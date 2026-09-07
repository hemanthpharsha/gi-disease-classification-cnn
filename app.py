import os
import numpy as np
import tensorflow as tf
from flask import Flask, abort, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import secrets
from datetime import datetime
import sqlite3
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_change_this_in_production_12345')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024
MINIMUM_CONFIDENCE_THRESHOLD = 0.15  # If max confidence is below this, likely not a GI image
# Temperature scaling calibrated from a labelled, held-out subset of the
# bundled GI-image dataset. A value above 1 reduces overconfident Softmax
# scores without changing the predicted class.
CONFIDENCE_TEMPERATURE = 1.535

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# A new value is created whenever the server starts.  Sessions from an earlier
# run are invalidated so reopening the project always begins at Medical Login.
SERVER_SESSION_ID = secrets.token_urlsafe(32)

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# SQLite does not understand NumPy scalar types.  Without this conversion, a
# ``numpy.float32`` confidence is stored as a BLOB instead of a REAL number.
# That breaks the percentage calculation when an older prediction is rendered
# on the dashboard.
def normalize_confidence(value):
    """Return a safe Python float for a prediction confidence value."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw_value = bytes(value)
        try:
            if len(raw_value) == np.dtype(np.float32).itemsize:
                value = np.frombuffer(raw_value, dtype=np.float32, count=1)[0]
            elif len(raw_value) == np.dtype(np.float64).itemsize:
                value = np.frombuffer(raw_value, dtype=np.float64, count=1)[0]
            else:
                return 0.0
        except ValueError:
            return 0.0

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return confidence if np.isfinite(confidence) else 0.0


def remove_uploaded_files(filenames):
    """Remove unreferenced uploads while keeping deletion inside the upload folder."""
    upload_root = os.path.abspath(app.config['UPLOAD_FOLDER'])

    for filename in set(filenames):
        if not filename:
            continue

        file_path = os.path.abspath(os.path.join(upload_root, filename))
        try:
            is_within_upload_root = os.path.commonpath([upload_root, file_path]) == upload_root
        except ValueError:
            is_within_upload_root = False

        if not is_within_upload_root:
            print(f"Skipped unsafe upload deletion path: {filename}")
            continue

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as e:
            # The database record is already removed; log the cleanup problem
            # without reporting a failed deletion to the administrator.
            print(f"Could not remove uploaded image {filename}: {e}")

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            is_valid_image INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Add profile fields for databases created before administrator profile
    # editing was available.
    user_columns = {
        column[1] for column in cursor.execute('PRAGMA table_info(users)').fetchall()
    }
    if 'full_name' not in user_columns:
        cursor.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    if 'mobile' not in user_columns:
        cursor.execute('ALTER TABLE users ADD COLUMN mobile TEXT')

    # Repair confidence values written by previous versions of the app.  Those
    # versions inserted NumPy float32 values directly, which SQLite persisted
    # as binary BLOBs.  Keeping this migration here makes existing accounts
    # work immediately after the application is restarted.
    legacy_confidences = cursor.execute(
        "SELECT id, confidence FROM predictions WHERE typeof(confidence) = 'blob'"
    ).fetchall()
    for prediction_id, confidence in legacy_confidences:
        cursor.execute(
            'UPDATE predictions SET confidence = ? WHERE id = ?',
            (normalize_confidence(confidence), prediction_id)
        )
    
    # Create default admin user
    admin_exists = cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin_exists:
        admin_hash = generate_password_hash('admin123')
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)',
            ('admin', 'admin@hospital.com', admin_hash, 'admin')
        )
    
    conn.commit()
    conn.close()

init_db()

# Define class names
class_names = [
    'dyed-lifted-polyps', 
    'dyed-resection-margins', 
    'esophagitis', 
    'normal-cecum', 
    'normal-pylorus', 
    'normal-z-line', 
    'polyps', 
    'ulcerative-colitis'
]
num_classes = len(class_names)

# Treatment recommendations
def get_treatment_recommendations(predicted_class):
    treatments = {
        "dyed-lifted-polyps": {
            "type": "Medical Treatment",
            "recommendations": [
                "Polyp Removal: Undergo endoscopic resection if the polyp poses a risk.",
                "Post-Procedure Care: Follow the doctor's advice regarding diet and medications.",
                "Regular Screening: Schedule follow-up endoscopies to monitor for recurrence or new polyps."
            ],
            "urgency": "Medium",
            "follow_up": "3-6 months",
            "specialist": "Gastroenterologist"
        },
        "normal-z-line": {
            "type": "No Treatment Needed",
            "recommendations": [
                "Healthy Lifestyle: Maintain a balanced diet to support overall digestive health.",
                "Routine Check-ups: Continue regular screenings as per your healthcare provider's advice.",
                "Stay Hydrated: Drink adequate water to maintain a healthy gastrointestinal tract."
            ],
            "urgency": "Low",
            "follow_up": "Annual screening",
            "specialist": "Primary Care Physician"
        },
        "polyps": {
            "type": "Medical Treatment",
            "recommendations": [
                "Endoscopic Removal: Remove polyps via colonoscopy to prevent potential malignancy.",
                "Lifestyle Adjustments: Avoid smoking, maintain a high-fiber diet, and reduce alcohol consumption.",
                "Periodic Monitoring: Schedule regular check-ups to detect new or recurring polyps early."
            ],
            "urgency": "Medium to High",
            "follow_up": "1-3 months",
            "specialist": "Gastroenterologist"
        },
        "dyed-resection-margins": {
            "type": "Medical Treatment",
            "recommendations": [
                "Post-Surgical Care: Follow prescribed antibiotics and pain relievers as needed.",
                "Healing Monitoring: Schedule follow-ups to ensure proper healing of resection sites.",
                "Lifestyle Guidance: Maintain a diet low in irritants to support gastrointestinal recovery."
            ],
            "urgency": "High",
            "follow_up": "2-4 weeks",
            "specialist": "Gastroenterologist/Surgeon"
        },
        "ulcerative-colitis": {
            "type": "Medical Treatment",
            "recommendations": [
                "Medication: Use anti-inflammatory drugs or immunosuppressants as prescribed.",
                "Dietary Adjustments: Follow a low-residue or anti-inflammatory diet to manage symptoms.",
                "Regular Monitoring: Attend follow-ups for symptom management and to prevent complications."
            ],
            "urgency": "High",
            "follow_up": "2-4 weeks",
            "specialist": "Gastroenterologist"
        },
        "esophagitis": {
            "type": "Medical Treatment",
            "recommendations": [
                "Medication: Use antacids, proton pump inhibitors, or other prescribed medications.",
                "Dietary Changes: Avoid acidic, spicy, or hot foods to reduce irritation.",
                "Lifestyle Adjustments: Stop smoking, avoid alcohol, and elevate the head while sleeping."
            ],
            "urgency": "Medium",
            "follow_up": "4-6 weeks",
            "specialist": "Gastroenterologist"
        },
        "normal-cecum": {
            "type": "No Treatment Needed",
            "recommendations": [
                "Routine Monitoring: Maintain regular health screenings as recommended.",
                "Healthy Eating: Focus on a balanced diet rich in fiber for optimal colon health.",
                "Stay Active: Engage in regular physical activity to promote gastrointestinal well-being."
            ],
            "urgency": "Low",
            "follow_up": "Annual screening",
            "specialist": "Primary Care Physician"
        },
        "normal-pylorus": {
            "type": "No Treatment Needed",
            "recommendations": [
                "Maintain Nutrition: Continue with a balanced diet and proper hydration.",
                "Routine Check-ups: Schedule regular health examinations as a preventive measure.",
                "Avoid Stomach Irritants: Limit spicy foods, caffeine, and alcohol for stomach health."
            ],
            "urgency": "Low",
            "follow_up": "Annual screening",
            "specialist": "Primary Care Physician"
        }
    }
    
    return treatments.get(predicted_class, {
        "type": "Consult Healthcare Provider",
        "recommendations": [
            "Please consult with a healthcare professional for proper diagnosis and treatment.",
            "Schedule an appointment with a gastroenterologist for detailed evaluation.",
            "Follow general health guidelines until professional consultation."
        ],
        "urgency": "Medium",
        "follow_up": "As soon as possible",
        "specialist": "Healthcare Professional"
    })

# Authentication decorator
@app.before_request
def invalidate_previous_server_sessions():
    """Require a new login after the Flask server has restarted."""
    if (
        session.get('user_id') is not None
        and session.get('server_session_id') != SERVER_SESSION_ID
    ):
        session.clear()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Model loading functions
def create_model_architecture():
    print("🔧 Creating model architecture...")
    
    base_model = ResNet101V2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling='max'
    )
    
    base_model.trainable = True
    
    model = Sequential([
        base_model,
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_model_safe():
    model = None
    
    possible_paths = [
        'final_gi_model.h5',
        'model/final_gi_model.h5',
        'final_gi_model/final_gi_model.h5',
        os.path.expanduser('~/Downloads/final_gi_model.h5'),
        os.path.expanduser('~/Desktop/final_gi_model.h5'),
        'final_gi_model.keras',
        'model/final_gi_model.keras'
    ]
    
    print("🔍 Searching for model file...")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Found model at: {path}")
            
            try:
                print(f"🔄 Method 1: Direct load_model()...")
                model = load_model(path)
                print(f"✅ Model loaded successfully with direct method!")
                return model, path
            except Exception as e:
                print(f"❌ Direct loading failed: {str(e)[:100]}...")
            
            try:
                print(f"🔄 Method 2: Architecture + weights loading...")
                model = create_model_architecture()
                
                model.compile(
                    optimizer=AdamW(learning_rate=3e-5, weight_decay=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                dummy_input = np.random.random((1, 224, 224, 3))
                _ = model(dummy_input)
                
                model.load_weights(path)
                print(f"✅ Model loaded successfully with weights method!")
                return model, path
                
            except Exception as e:
                print(f"❌ Weights loading failed: {str(e)[:100]}...")
    
    return None, None

# Load model
print("🚀 Loading AI model...")
model, model_path = load_model_safe()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_gi_image(probabilities, max_confidence):
    """
    Check if the image is likely a valid gastrointestinal image
    Returns: (is_valid, reason)
    """
    # If the maximum confidence is very low, it's likely not a GI image
    if max_confidence < MINIMUM_CONFIDENCE_THRESHOLD:
        return False, "Image does not appear to be a gastrointestinal endoscopy image. Please upload a valid medical image."
    
    # Check if probabilities are too uniform (indicates uncertainty)
    prob_std = np.std(probabilities)
    if prob_std < 0.05:  # Very uniform distribution
        return False, "Unable to classify image. Please ensure you upload a clear gastrointestinal endoscopy image."
    
    return True, ""


def calibrate_probabilities(probabilities):
    """Apply temperature scaling to an existing probability distribution."""
    safe_probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0)
    scaled_log_probabilities = np.log(safe_probabilities) / CONFIDENCE_TEMPERATURE
    scaled_log_probabilities -= np.max(scaled_log_probabilities)
    scaled_probabilities = np.exp(scaled_log_probabilities)
    return scaled_probabilities / scaled_probabilities.sum()

def predict_image(img_path):
    if model is None:
        return None, None, None, None, None, False, "Model not loaded"
    
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_output = np.asarray(model.predict(img_array, verbose=0)[0], dtype=np.float32)

        # The classifier ends with a softmax layer, so its normal output is
        # already a probability distribution. Applying softmax again turns a
        # high confidence (for example 95%) into roughly 28% for eight classes.
        # Keep probability output unchanged, while still supporting a model
        # that returns raw logits.
        if np.all(model_output >= 0) and np.isclose(model_output.sum(), 1.0, atol=1e-4):
            raw_probabilities = model_output
        else:
            raw_probabilities = tf.nn.softmax(model_output).numpy()

        probabilities = calibrate_probabilities(raw_probabilities)
        
        pred_class_idx = np.argmax(probabilities)
        pred_class = class_names[pred_class_idx]
        original_confidence = float(probabilities[pred_class_idx])
        
        # Validate if it's a GI image
        is_valid, error_message = is_valid_gi_image(probabilities, original_confidence)
        
        if not is_valid:
            return None, None, None, None, None, False, error_message
        
        confidence_percent = original_confidence * 100
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(img)
        ax1.set_title(
            f"Predicted: {pred_class.replace('-', ' ').title()}\n"
            f"Calibrated confidence: {confidence_percent:.2f}%",
            fontsize=12
        )
        ax1.axis('off')
        
        colors = ['green' if i == pred_class_idx else 'lightblue' for i in range(len(class_names))]
        bars = ax2.barh(range(len(class_names)), probabilities, color=colors)
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels([name.replace('-', ' ').title() for name in class_names], fontsize=10)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{prob:.6f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return (pred_class, 
                original_confidence, 
                confidence_percent,
                plot_url, 
                dict(zip(class_names, probabilities)),
                True,
                "")
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, False, f"Error processing image: {str(e)}"

# --- Authentication Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please enter both username and password.', 'warning')
            return render_template('login.html')
        
        try:
            conn = sqlite3.connect('users.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            user = cursor.execute(
                'SELECT * FROM users WHERE username = ? OR email = ?', 
                (username, username)
            ).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                # Update last login
                cursor.execute(
                    'UPDATE users SET last_login = ? WHERE id = ?',
                    (datetime.now(), user['id'])
                )
                conn.commit()
                
                # Set session data
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['logged_in'] = True
                session['server_session_id'] = SERVER_SESSION_ID
                
                conn.close()
                
                flash(f'Welcome back, {user["username"]}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                conn.close()
                flash('Invalid username or password. Please try again.', 'danger')
                return render_template('login.html')
                
        except Exception as e:
            print(f"Login error: {e}")
            flash('An error occurred. Please try again.', 'danger')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirmPassword', '')
        
        if not username or not email or not password:
            flash('All fields are required.', 'warning')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'warning')
            return render_template('register.html')
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # Check if user exists
            existing_user = cursor.execute(
                'SELECT * FROM users WHERE username = ? OR email = ?', 
                (username, email)
            ).fetchone()
            
            if existing_user:
                flash('Username or email already exists. Please choose another.', 'warning')
                conn.close()
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login with your credentials.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            print(f"Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')
    
    # This renders the new 'register.html' file
    return render_template('register.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'User')
    # Clear all session data
    session.clear()
    flash(f'Goodbye, {username}! You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

# --- Application Routes ---

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user's recent predictions
        predictions = cursor.execute(
            'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
            (session['user_id'],)
        ).fetchall()
        predictions = [dict(prediction) for prediction in predictions]
        for prediction in predictions:
            prediction['confidence'] = normalize_confidence(prediction['confidence'])
        
        # Get statistics for dashboard
        total_predictions = cursor.execute(
            'SELECT COUNT(*) FROM predictions WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()[0]
        
        valid_predictions = cursor.execute(
            'SELECT COUNT(*) FROM predictions WHERE user_id = ? AND is_valid_image = 1',
            (session['user_id'],)
        ).fetchone()[0]
        
        conn.close()
        print(f"Dashboard - User ID: {session.get('user_id')}")
        print(f"Dashboard - Total Predictions: {total_predictions}")
        print(f"Dashboard - Valid Predictions: {valid_predictions}")
        print(f"Dashboard - Predictions Found: {len(predictions) if predictions else 0}")
        return render_template('dashboard.html', 
                               predictions=predictions,
                               total_predictions=total_predictions,
                               valid_predictions=valid_predictions)
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash('Error loading dashboard.', 'danger')
        return redirect(url_for('upload_file'))

# This route redirects the root URL '/' to the login page
@app.route('/')
def index():
    return redirect(url_for('login'))

# This is your main upload and prediction page
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if model is None:
        flash('AI model is not loaded. Please contact administrator.', 'danger')
        return render_template('index.html', error_message='AI model is not loaded.')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected. Please choose an image file.', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image file.', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_image(filepath)
            pred_class, original_confidence, confidence_percent, plot_url, probabilities, is_valid, error_msg = result
            
            if not is_valid:
                # Save failed prediction
                conn = sqlite3.connect('users.db')
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO predictions (user_id, filename, prediction, confidence, is_valid_image) VALUES (?, ?, ?, ?, ?)',
                    (session['user_id'], filename, 'Invalid Image', 0, 0)
                )
                conn.commit()
                conn.close()
                
                flash(error_msg, 'danger')
                return render_template('index.html', error_message=error_msg)
            
            # Save prediction to database
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO predictions (user_id, filename, prediction, confidence, is_valid_image) VALUES (?, ?, ?, ?, ?)',
                (session['user_id'], filename, pred_class, float(original_confidence), 1)
            )
            conn.commit()
            conn.close()
            
            # Get treatment recommendations
            treatment_info = get_treatment_recommendations(pred_class)
            
            # This renders 'index.html' (the main upload page) with the results
            return render_template('index.html', 
                                   filename=filename,
                                   prediction=pred_class,
                                   original_confidence=f"{original_confidence:.2%}",
                                   confidence=f"{confidence_percent:.2f}%",
                                   confidence_percent=confidence_percent,
                                   plot_url=plot_url,
                                   probabilities=probabilities,
                                   treatment_info=treatment_info)
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.', 'warning')
            return redirect(request.url)
    
    # This renders 'index.html' (the main upload page) for a GET request
    return render_template('index.html')

@app.route('/admin')
@admin_required
def admin_panel():
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        search_query = request.args.get('search', '').strip()
        if search_query:
            users = cursor.execute('''
                SELECT * FROM users
                WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE
                ORDER BY created_at DESC
            ''', (search_query, search_query)).fetchall()
        else:
            users = cursor.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
        conn.close()
        
        return render_template('admin.html', users=users, search_query=search_query)
    except Exception as e:
        print(f"Admin panel error: {e}")
        flash('Error loading admin panel.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/admin/profile', methods=['GET', 'POST'])
@admin_required
def admin_profile():
    """Allow the signed-in administrator to update their profile."""
    conn = None
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        admin = cursor.execute(
            'SELECT * FROM users WHERE id = ?', (session['user_id'],)
        ).fetchone()

        if admin is None or admin['role'] != 'admin':
            session.clear()
            flash('Please log in with an administrator account.', 'warning')
            return redirect(url_for('login'))

        if request.method == 'POST':
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip()
            mobile = request.form.get('mobile', '').strip()
            current_password = request.form.get('current_password', '')
            new_password = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')

            if not email:
                flash('Email is required.', 'warning')
                return render_template('admin_profile.html', admin=admin)

            if mobile and not all(character.isdigit() or character in ' +-()' for character in mobile):
                flash('Enter a valid mobile number.', 'warning')
                return render_template('admin_profile.html', admin=admin)

            email_in_use = cursor.execute(
                'SELECT id FROM users WHERE email = ? COLLATE NOCASE AND id != ?',
                (email, admin['id'])
            ).fetchone()
            if email_in_use:
                flash('That email address is already in use.', 'warning')
                return render_template('admin_profile.html', admin=admin)

            password_hash = admin['password_hash']
            if new_password or confirm_password or current_password:
                if not current_password or not check_password_hash(password_hash, current_password):
                    flash('Enter your current password to make a password change.', 'warning')
                    return render_template('admin_profile.html', admin=admin)
                if len(new_password) < 6:
                    flash('New password must be at least 6 characters long.', 'warning')
                    return render_template('admin_profile.html', admin=admin)
                if new_password != confirm_password:
                    flash('New passwords do not match.', 'warning')
                    return render_template('admin_profile.html', admin=admin)
                password_hash = generate_password_hash(new_password)

            cursor.execute('''
                UPDATE users
                SET full_name = ?, email = ?, mobile = ?, password_hash = ?
                WHERE id = ?
            ''', (full_name, email, mobile, password_hash, admin['id']))
            conn.commit()

            flash('Administrator profile updated successfully.', 'success')
            return redirect(url_for('admin_profile'))

        return render_template('admin_profile.html', admin=admin)
    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
        print(f"Admin profile error: {e}")
        flash('Unable to update the administrator profile.', 'danger')
        return redirect(url_for('admin_panel'))
    finally:
        if conn is not None:
            conn.close()


@app.route('/admin/users/<int:user_id>/predictions')
@admin_required
def admin_user_predictions(user_id):
    """Show one selected user's prediction history to an administrator."""
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        user = cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user is None:
            conn.close()
            flash('User not found.', 'warning')
            return redirect(url_for('admin_panel'))

        predictions = cursor.execute('''
            SELECT * FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,)).fetchall()
        conn.close()

        predictions = [dict(prediction) for prediction in predictions]
        for prediction in predictions:
            prediction['confidence'] = normalize_confidence(prediction['confidence'])

        return render_template(
            'user_predictions.html',
            selected_user=user,
            predictions=predictions
        )
    except sqlite3.Error as e:
        print(f"User prediction history error: {e}")
        flash('Unable to load the user prediction history.', 'danger')
        return redirect(url_for('admin_panel'))


@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete a non-admin user, their prediction history, and unused uploads."""
    conn = None
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        user = cursor.execute(
            'SELECT id, username, role FROM users WHERE id = ?', (user_id,)
        ).fetchone()
        if user is None:
            return jsonify({'success': False, 'error': 'User not found.'}), 404

        # Keep administrator accounts protected so this operation cannot lock
        # the application out of administrative access.
        if user['role'] == 'admin':
            return jsonify({
                'success': False,
                'error': 'Administrator accounts cannot be deleted.'
            }), 403

        filenames = {
            row['filename'] for row in cursor.execute(
                'SELECT filename FROM predictions WHERE user_id = ?', (user_id,)
            ).fetchall()
            if row['filename']
        }

        cursor.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))

        # A filename is normally unique, but retain it if another prediction
        # still references it.
        remaining_filenames = set()
        if filenames:
            placeholders = ', '.join('?' for _ in filenames)
            remaining_filenames = {
                row['filename'] for row in cursor.execute(
                    f'SELECT DISTINCT filename FROM predictions WHERE filename IN ({placeholders})',
                    tuple(filenames)
                ).fetchall()
            }

        conn.commit()
        conn.close()
        conn = None

        remove_uploaded_files(filenames - remaining_filenames)
        return jsonify({
            'success': True,
            'message': f"User '{user['username']}' and their saved predictions were deleted."
        })
    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
        print(f"Delete user error: {e}")
        return jsonify({'success': False, 'error': 'Unable to delete the user.'}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/uploaded_file/<filename>')
@login_required
def uploaded_file(filename):
    # Only the prediction owner (or an admin) may view an uploaded image.
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    prediction = conn.execute(
        'SELECT user_id FROM predictions WHERE filename = ? ORDER BY id DESC LIMIT 1',
        (filename,)
    ).fetchone()
    conn.close()

    if prediction is None:
        abort(404)
    if session.get('role') != 'admin' and prediction['user_id'] != session['user_id']:
        abort(403)

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analysis/<int:analysis_id>', methods=['GET', 'POST'])
@login_required
def analysis_details(analysis_id):
    """Show a saved prediction with its treatment recommendations."""
    # Details is a read-only page.  Accept POST from older browser pages and
    # redirect it to the normal GET URL instead of showing a 405 error.
    if request.method == 'POST':
        return redirect(url_for('analysis_details', analysis_id=analysis_id), code=303)

    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row

        query = 'SELECT * FROM predictions WHERE id = ?'
        params = [analysis_id]
        if session.get('role') != 'admin':
            query += ' AND user_id = ?'
            params.append(session['user_id'])

        prediction = conn.execute(query, params).fetchone()
        conn.close()

        if prediction is None:
            flash('That saved prediction was not found or you do not have permission to view it.', 'warning')
            return redirect(url_for('dashboard'))

        if prediction['is_valid_image'] != 1:
            flash('Treatment recommendations are unavailable for an invalid image.', 'warning')
            return redirect(url_for('dashboard'))

        original_confidence = normalize_confidence(prediction['confidence'])
        confidence_percent = original_confidence * 100
        predicted_class = prediction['prediction']

        return render_template(
            'index.html',
            filename=prediction['filename'],
            prediction=predicted_class,
            original_confidence=f'{original_confidence:.2%}',
            confidence=f'{confidence_percent:.4f}%',
            confidence_percent=confidence_percent,
            treatment_info=get_treatment_recommendations(predicted_class)
        )
    except Exception as e:
        print(f"Analysis details error: {e}")
        flash('Unable to load the saved prediction details.', 'danger')
        return redirect(url_for('dashboard'))

# --- Utility Routes ---

@app.route('/debug')
@login_required
def debug_info():
    import tensorflow as tf
    
    debug_info = {
        'model_loaded': model is not None,
        'model_path': model_path if model_path else 'Not found',
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'class_names': class_names,
        'num_classes': num_classes,
        'confidence_temperature': CONFIDENCE_TEMPERATURE,
        'confidence_threshold': MINIMUM_CONFIDENCE_THRESHOLD,
        'current_user': session.get('username', 'Anonymous'),
        'session_data': dict(session)
    }
    
    if model is not None:
        debug_info['model_input_shape'] = str(model.input_shape)
        debug_info['model_output_shape'] = str(model.output_shape)
    
    return jsonify(debug_info)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'model_not_loaded',
        'model_loaded': model is not None,
        'classes': class_names
    })

@app.route('/delete_analysis/<int:analysis_id>', methods=['DELETE'])
@login_required
def delete_analysis(analysis_id):
    conn = None
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if the analysis belongs to the current user OR if user is admin
        query = 'SELECT * FROM predictions WHERE id = ?'
        params = (analysis_id,)
        
        if session.get('role') != 'admin':
            query += ' AND user_id = ?'
            params = (analysis_id, session['user_id'])
            
        analysis = cursor.execute(query, params).fetchone()
        
        if analysis:
            cursor.execute('DELETE FROM predictions WHERE id = ?', (analysis_id,))
            image_is_still_used = cursor.execute(
                'SELECT 1 FROM predictions WHERE filename = ? LIMIT 1',
                (analysis['filename'],)
            ).fetchone()
            conn.commit()
            conn.close()
            conn = None

            if not image_is_still_used:
                remove_uploaded_files([analysis['filename']])
            return jsonify({'success': True})
        else:
            conn.close()
            conn = None
            return jsonify({'success': False, 'error': 'Analysis not found or unauthorized'}), 404

    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
        print(f"Delete analysis error: {e}")
        return jsonify({'success': False, 'error': 'Unable to delete the saved prediction.'}), 500
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ENHANCED FLASK APP STATUS")
    print("="*50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🎯 Classes: {len(class_names)} classes")
    print(f"🤖 Model loaded: {'✅ YES' if model else '❌ NO'}")
    print(f"🔐 Authentication: ✅ ENABLED")
    print(f"✅ Image validation: ENABLED (threshold: {MINIMUM_CONFIDENCE_THRESHOLD})")
    print(f"💊 Treatment recommendations: ✅ ENABLED")
    print(f"👤 Default admin: username='admin', password='admin123'")
    
    if model is None:
        print("\n❗ MODEL NOT LOADED - TROUBLESHOOTING:")
        print("1. Check if final_gi_model.h5 exists in current directory")
        print("2. Try downloading model again from Colab")
        print("3. Visit http://localhost:5000/debug for detailed info")
    
    print(f"\n🌐 Starting server on http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
