import http.client
import os
import json
import secrets
import string
from datetime import datetime, timedelta
from io import BytesIO
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import click
from flask.cli import with_appcontext

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///reevuee.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Email Configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'REEvuee Support <help1.reevuee@gmail.com>')

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# Constants
MAX_REVIEWS = 20
PASSWORD_MIN_LENGTH = 8

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    reset_token = db.Column(db.String(100))
    reset_token_expiration = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('AnalysisHistory', backref='user', lazy=True, cascade='all, delete-orphan')

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    review_text = db.Column(db.Text)
    prediction = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    product_id = db.Column(db.String(100))
    product_name = db.Column(db.String(200))
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_type = db.Column(db.String(50))  # 'single' or 'product'
    analysis_platform = db.Column(db.String(50))  # 'flipkart' or 'amazon'

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# CLI Commands
@app.cli.command("create-admin")
@with_appcontext
def create_admin():
    """Create the admin user"""
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@reevuee.com',
            password=generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'Admin@123'))
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created")
    else:
        print("Admin user already exists")

# Helper Functions
def generate_reset_token():
    """Generate a secure random token for password reset"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(32))

def send_reset_email(user):
    """Send password reset email to user"""
    try:
        token = generate_reset_token()
        user.reset_token = token
        user.reset_token_expiration = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()
        
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = user.email
        msg['Subject'] = 'REEvuee - Password Reset Request'
        
        reset_url = url_for('reset_password', token=token, _external=True)
        
        html = f"""<html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #00b4d8;">Password Reset Request</h2>
            <p>Hello,</p>
            <p>We received a request to reset your REEvuee account password.</p>
            <p>Click the button below to reset your password:</p>
            <p><a href="{reset_url}" style="background-color: #00b4d8; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Reset Password</a></p>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, please ignore this email.</p>
            <p>Thanks,<br>The REEvuee Team</p>
        </body>
        </html>"""
        
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        db.session.rollback()
        return False

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def initialize_model():
    try:
        data = pd.read_csv('Fake-Reviews-Detection-main/reviews_dataset.csv')
        df = pd.DataFrame(data)
        df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
        df['Numeric_Label'] = df['Label'].map({'OR': 1, 'CG': 0})

        X_train, X_test, y_train, y_test = train_test_split(
            df['Cleaned_Review'], df['Numeric_Label'], test_size=0.2, random_state=42)

        tfidf = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

        return tfidf, model, accuracy
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None, 0

# Initialize model and sentiment analyzer
tfidf, model, accuracy = initialize_model()
analyzer = SentimentIntensityAnalyzer()

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        flash('Please fill in all fields', 'error')
        return redirect(url_for('home'))
    
    user = User.query.filter_by(username=username).first()
    
    if user and check_password_hash(user.password, password):
        login_user(user)
        return redirect(url_for('analyze'))
    
    flash('Invalid username or password', 'error')
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('signup'))
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
            
        if len(password) < PASSWORD_MIN_LENGTH:
            flash(f'Password must be at least {PASSWORD_MIN_LENGTH} characters', 'error')
            return redirect(url_for('signup'))
            
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('analyze'))
    
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email address', 'error')
            return redirect(url_for('forgot_password'))
            
        user = User.query.filter_by(email=email).first()
        if user:
            if send_reset_email(user):
                flash('Password reset link has been sent to your email', 'info')
            else:
                flash('Error sending reset email. Please try again later or contact support.', 'error')
            return redirect(url_for('forgot_password'))
        
        flash('If an account exists with that email, a reset link has been sent', 'info')
        return redirect(url_for('forgot_password'))
    
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or user.reset_token_expiration < datetime.utcnow():
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('reset_password', token=token))
            
        if len(password) < PASSWORD_MIN_LENGTH:
            flash(f'Password must be at least {PASSWORD_MIN_LENGTH} characters', 'error')
            return redirect(url_for('reset_password', token=token))
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('reset_password', token=token))
        
        user.password = generate_password_hash(password)
        user.reset_token = None
        user.reset_token_expiration = None
        db.session.commit()
        
        flash('Your password has been updated! Please log in.', 'success')
        return redirect(url_for('home'))
    
    return render_template('reset_password.html', token=token)

@app.route('/analyze')
@login_required
def analyze():
    return render_template('analyze.html', 
                         accuracy=round(accuracy * 100, 2),
                         logged_in=current_user.is_authenticated)

@app.route('/about')
def about():
    return render_template('about.html', logged_in=current_user.is_authenticated)

@app.route('/contact')
def contact():
    return render_template('contact.html', logged_in=current_user.is_authenticated)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    review = request.form.get('review', '').strip()
    if not review:
        flash('Please enter a review', 'error')
        return redirect(url_for('analyze'))

    try:
        cleaned = preprocess_text(review)
        features = tfidf.transform([cleaned])
        prediction = 'Real' if model.predict(features)[0] == 1 else 'Fake'
        emotion = detect_emotion(review)
        confidence = get_prediction_confidence(features)
        sentiment_scores = analyzer.polarity_scores(review)
        
        # Generate visualization
        viz = generate_single_review_visualization({
            'prediction': prediction,
            'sentiment': emotion,
            'confidence': confidence,
            'sentiment_scores': sentiment_scores
        })
        
        # Generate recommendation
        recommendation = generate_single_review_recommendation({
            'prediction': prediction,
            'sentiment': emotion,
            'confidence': confidence,
            'sentiment_scores': sentiment_scores
        })
        
        # Save to history
        history = AnalysisHistory(
            user_id=current_user.id,
            review_text=review,
            prediction=prediction,
            sentiment=emotion,
            analysis_type='single'
        )
        db.session.add(history)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash('Error analyzing review', 'error')
        return redirect(url_for('analyze'))

    return render_template('analyze.html',
                         review=review,
                         result=prediction,
                         emotion=emotion,
                         visualization=viz,
                         recommendation=recommendation,
                         accuracy=round(accuracy * 100, 2),
                         logged_in=current_user.is_authenticated)

@app.route('/analyze_product', methods=['POST'])
@login_required
def analyze_product():
    pid = request.form.get('product_pid', '').strip()
    if not pid:
        flash('Please enter a Flipkart product PID', 'error')
        return redirect(url_for('analyze'))

    reviews, product_name = fetch_flipkart_reviews_api(pid)
    if not reviews:
        flash('No reviews found or Flipkart API failed', 'error')
        return redirect(url_for('analyze'))

    return analyze_reviews_and_render(reviews, product_name, 'flipkart', pid)

@app.route('/analyze_amazon', methods=['POST'])
@login_required
def analyze_amazon():
    asin = request.form.get('product_asin', '').strip()
    if not asin:
        flash('Please enter an Amazon ASIN', 'error')
        return redirect(url_for('analyze'))

    reviews, product_name = fetch_amazon_reviews_api(asin)
    if not reviews:
        flash('No reviews found or Amazon API failed', 'error')
        return redirect(url_for('analyze'))

    return analyze_reviews_and_render(reviews, product_name, 'amazon', asin)

def analyze_reviews_and_render(reviews, product_name, platform, product_id):
    analyzed = []
    for review in reviews:
        try:
            cleaned = preprocess_text(review)
            features = tfidf.transform([cleaned])
            prediction = 'Real' if model.predict(features)[0] == 1 else 'Fake'
            emotion = detect_emotion(review)
            analyzed.append({'Review': review, 'Prediction': prediction, 'Sentiment': emotion})
        except Exception as e:
            print(f"Error analyzing review: {e}")
            continue

    if not analyzed:
        flash('Error analyzing reviews', 'error')
        return redirect(url_for('analyze'))

    df = pd.DataFrame(analyzed)
    viz = generate_visualization(df)
    recommendation = generate_recommendation(df)
    samples = df.sample(min(3, len(df))).to_dict('records')
    
    history = AnalysisHistory(
        user_id=current_user.id,
        product_id=product_id,
        product_name=product_name,
        prediction=f"{len([r for r in analyzed if r['Prediction'] == 'Real'])}/{len(analyzed)} Real",
        sentiment=", ".join(set(r['Sentiment'] for r in analyzed)),
        analysis_type='product',
        analysis_platform=platform
    )
    db.session.add(history)
    db.session.commit()

    return render_template('product_analysis.html',
                         visualization=viz,
                         recommendation=recommendation,
                         sample_reviews=samples,
                         total_reviews=len(df),
                         product_name=product_name,
                         product_id=product_id,
                         logged_in=current_user.is_authenticated)

@app.route('/history')
@login_required
def history():
    user_history = AnalysisHistory.query.filter_by(user_id=current_user.id)\
                          .order_by(AnalysisHistory.analysis_date.desc())\
                          .limit(50)\
                          .all()
    return render_template('history.html', history=user_history, logged_in=current_user.is_authenticated)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    db.session.rollback()
    return render_template('500.html'), 500

# API Helper Functions
def fetch_flipkart_reviews_api(pid):
    try:
        conn = http.client.HTTPSConnection("real-time-flipkart-api.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': os.environ.get('FLIPKART_API_KEY'),
            'x-rapidapi-host': "real-time-flipkart-api.p.rapidapi.com"
        }
        url = f"/product-details?pid={pid}&pincode=110011"
        conn.request("GET", url, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        parsed = json.loads(data)

        reviews_raw = parsed.get("productReviews", [])
        reviews = [review.get("text", "") for review in reviews_raw if review.get("text")]
        return reviews[:MAX_REVIEWS], parsed.get("productName", "Unknown Product")
    except Exception as e:
        print(f"Flipkart API failed: {e}")
        return [], "Unknown Product"

def fetch_amazon_reviews_api(asin):
    try:
        conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': os.environ.get('AMAZON_API_KEY'),
            'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
        }
        conn.request("GET", f"/product-offers?asin={asin}&country=US&limit=100&page=1", headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        parsed = json.loads(data)

        reviews_raw = parsed.get("reviews", [])
        reviews = [review.get("text", "") for review in reviews_raw if review.get("text")]
        return reviews[:MAX_REVIEWS], parsed.get("title", "Unknown Product")
    except Exception as e:
        print(f"Amazon API failed: {e}")
        return [], "Unknown Product"

def detect_emotion(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
def get_prediction_confidence(features):
    """Get the confidence score of the prediction"""
    try:
        proba = model.predict_proba(features)[0]
        return round(max(proba) * 100, 2)
    except:
        return None

def generate_single_review_visualization(analysis):
    """Generate visualization for single review analysis"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plt.subplots_adjust(wspace=0.3)
        
        # 1. Prediction with confidence
        pred_label = analysis['prediction']
        confidence = analysis.get('confidence', 0)
        
        # Define colors based on prediction
        if pred_label == 'Real':
            colors = ['#28a745', '#f8f9fa']  # Green for Real, light gray for remaining
        else:
            colors = ['#dc3545', '#f8f9fa']   # Red for Fake, light gray for remaining
            
        axes[0].pie([confidence, 100-confidence], 
                   labels=[f'{pred_label}\n{confidence}%', ''], 
                   colors=colors,
                   startangle=90)
        axes[0].set_title('Authenticity Confidence')
        
        # 2. Sentiment breakdown
        scores = analysis['sentiment_scores']
        sentiment_data = {
            'Positive': scores['pos'],
            'Negative': scores['neg'],
            'Neutral': scores['neu']
        }
        axes[1].pie(sentiment_data.values(), labels=sentiment_data.keys(),
                   colors=['#28a745', '#dc3545', '#6c757d'],
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Sentiment Composition')
        
        # 3. Sentiment scores bar chart
        sentiment_types = ['Negative', 'Neutral', 'Positive', 'Compound']
        sentiment_values = [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
        colors = ['#dc3545', '#6c757d', '#28a745', '#007bff']
        axes[2].bar(sentiment_types, sentiment_values, color=colors)
        axes[2].set_ylim(0, 1)
        axes[2].set_title('Detailed Sentiment Scores')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        print(f"Error generating single review visualization: {e}")
        return None

def generate_single_review_recommendation(analysis):
    """Generate detailed recommendation for single review"""
    prediction = analysis['prediction']
    sentiment = analysis['sentiment']
    confidence = analysis.get('confidence', 0)
    compound_score = analysis['sentiment_scores']['compound']
    
    if prediction == 'Fake':
        if confidence > 80:
            return "⚠️ Strong Warning: This review is very likely fake ({}% confidence). Be extremely cautious.".format(confidence)
        else:
            return "⚠️ Warning: This review appears to be fake ({}% confidence). Be cautious.".format(confidence)
    
    # For real reviews
    if sentiment == 'Positive':
        if compound_score > 0.5:
            return "✅ Strong Positive: This is a highly trustworthy positive review (score: {:.2f}).".format(compound_score)
        else:
            return "👍 Positive: This appears to be a genuine positive review (score: {:.2f}).".format(compound_score)
    elif sentiment == 'Negative':
        if compound_score < -0.5:
            return "❌ Strong Negative: This is a strongly negative but genuine review (score: {:.2f}).".format(compound_score)
        else:
            return "⚠️ Negative: This appears to be a legitimate negative review (score: {:.2f}).".format(compound_score)
    else:
        return "🤔 Neutral: This appears to be a genuine but neutral review (score: {:.2f}).".format(compound_score)
#---------------------------------------------------------------

#---------------------------------------------------------------



def generate_visualization(reviews_df):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        fake_real_counts = reviews_df['Prediction'].value_counts()
        axes[0, 0].pie(fake_real_counts, labels=fake_real_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Fake vs Real Reviews')

        sentiment_counts = reviews_df['Sentiment'].value_counts()
        axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
        axes[0, 1].set_title('Sentiment Distribution')

        if 'Fake' in fake_real_counts:
            fake_sentiment = reviews_df[reviews_df['Prediction'] == 'Fake']['Sentiment'].value_counts()
            axes[1, 0].pie(fake_sentiment, labels=fake_sentiment.index, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Fake Reviews Sentiment')

        if 'Real' in fake_real_counts:
            real_sentiment = reviews_df[reviews_df['Prediction'] == 'Real']['Sentiment'].value_counts()
            axes[1, 1].pie(real_sentiment, labels=real_sentiment.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Real Reviews Sentiment')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return None

def generate_recommendation(df):
    total = len(df)
    if total == 0:
        return "No reviews available for analysis."

    real_pct = len(df[df['Prediction'] == 'Real']) / total * 100
    positive_pct = len(df[df['Sentiment'] == 'Positive']) / total * 100

    if real_pct < 50:
        return "⚠️ Warning: Most reviews appear fake. Be cautious."
    elif positive_pct < 30:
        return "❌ Not Recommended: Mostly negative reviews."
    elif positive_pct > 70 and real_pct > 70:
        return "✅ Highly Recommended: Genuine positive reviews."
    elif positive_pct > 50:
        return "👍 Recommended: Mostly positive reviews."
    else:
        return "🤔 Mixed Reviews: Consider carefully."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))