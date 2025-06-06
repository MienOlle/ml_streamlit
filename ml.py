import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import emoji
import contractions
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from scipy.sparse import hstack
from scipy.stats import pearsonr
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor, LGBMClassifier

# lightbgm
try:
    
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

st.set_page_config(
    page_title="Emotion Detection & Intensity Analysis",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Emotion Detection & Intensity Analysis 🎭")
st.markdown("Analyze text for emotional content and intensity using machine learning models")

# config smping
st.sidebar.header("⚙️ Configuration")

# data split
test_size = st.sidebar.slider(
    "Test Set Size (%)", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=5,
    help="Percentage of data to use for testing"
) / 100

# model choosing
regression_models = ["Ridge", "LightGBM", "Ensemble (Ridge + LightGBM)"]
if not LIGHTGBM_AVAILABLE:
    regression_models = ["Ridge"]
    st.sidebar.warning("LightGBM not available. Install with: pip install lightgbm")

classification_models = ["Logistic Regression", "LightGBM", "SVM", "Ensemble (LR + LightGBM)"]
if not LIGHTGBM_AVAILABLE:
    classification_models = ["Logistic Regression", "SVM"]

selected_reg_model = st.sidebar.selectbox(
    "Regression Model",
    regression_models,
    index=len(regression_models)-1 if LIGHTGBM_AVAILABLE else 0,
    help="Model for predicting emotion intensity"
)

selected_clf_model = st.sidebar.selectbox(
    "Classification Model", 
    classification_models,
    index=len(classification_models)-1 if LIGHTGBM_AVAILABLE else 0,
    help="Model for predicting emotion category"
)

# alpha reg
alpha = 0.3
if "Ensemble" in selected_reg_model:
    alpha = st.sidebar.slider(
        "Ensemble Alpha (Ridge weight) - For Regression",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Weight for Ridge model in ensemble (1-alpha for LightGBM)"
    )

#text preprocess
def convert_emojis(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r':([a-zA-Z_]+):', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    # Lowercase
    text = text.lower()
    # expand contractions
    text = contractions.fix(text)
    # convert emojis
    text = convert_emojis(text)
    # remoive urls
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # remove user text aneh2
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"[^a-zA-Z\s.,!?']", '', text)
    # remove extra space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# load datasets
@st.cache_data
def load_datasets():
    """Load your real datasets"""
    try:
        train = pd.read_csv('train.csv', delimiter='\t')
        test = pd.read_csv('test.csv', delimiter='\t')
        dev = pd.read_csv('dev.csv', delimiter='\t')
        
        # dsiplay data detail sedikit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train Set", f"{len(train)} samples")
        with col2:
            st.metric("Test Set", f"{len(test)} samples")
        with col3:
            st.metric("Dev Set", f"{len(dev)} samples")
        
        # combine dtasets
        combined_df = pd.concat([train, dev, test], ignore_index=True)
        
        # preview combined
        with st.expander("📊 Data Preview"):
            st.write("**Combined Dataset Columns:**", list(combined_df.columns))
            st.write("**First 5 rows:**")
            st.dataframe(combined_df.head())
            st.write("**Dataset Shape:**", combined_df.shape)
        
        return combined_df
        
    except FileNotFoundError as e:
        st.error(f"❌ Dataset files not found: {str(e)}")
        st.error("Please ensure these files are in the same directory as your app:")
        st.code("- train.csv\n- test.csv\n- dev.csv")
        return None
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None

# lexicon
@st.cache_data
def load_lexicons():
    """Load all lexicon files"""
    lexicons = {}
    
    # Load EmoLex
    def load_lex(filepath):
        lexicon = defaultdict(dict)
        with open(filepath, 'r') as file:
            for line in file:
                word, emotion, value = line.strip().split('\t')
                if int(value) == 1:
                    lexicon[word][emotion] = 1
        return lexicon
    
    lexicons['nrc'] = load_lex("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

# Load VAD Lexicon
    def load_nrc_vad(filepath):
        vad_lex = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                word, val, aro, dom = line.strip().split('\t')
                vad_lex[word] = {
                    'valence': float(val),
                    'arousal': float(aro),
                    'dominance': float(dom)
                }
        return vad_lex
    
    lexicons['vad'] = load_nrc_vad("NRC-VAD-Lexicon-v2.1.txt")

    def load_nrc_hash_emo(filepath):
        lexicon = defaultdict(dict)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                emotion, word, score = line.strip().split('\t')
                lexicon[word][emotion] = float(score)
        return lexicon
    
    lexicons['hash'] = load_nrc_hash_emo('NRC-Hashtag-Emotion-Lexicon-v0.2.txt')
    return lexicons

# feature extract
def extract_lex(text, lexicon):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'sadness', 'surprise', 'trust', 'positive', 'negative']
    counts = dict.fromkeys(emotions, 0)

    for word in text.split():
        if word in lexicon:
            for emo in lexicon[word]:
                if emo in counts:
                    counts[emo] += 1
    return [counts[emo] for emo in emotions]

def extract_vad(text, lexicon):
    valence = []
    arousal = []
    dominance = []

    for word in text.split():
        if word in lexicon:
            valence.append(lexicon[word]['valence'])
            arousal.append(lexicon[word]['arousal'])
            dominance.append(lexicon[word]['dominance'])

    # return 0 jika gada word match
    if not valence:
        return [0.0, 0.0, 0.0]

    # kalo ada return mean
    return [
        np.mean(valence),
        np.mean(arousal),
        np.mean(dominance)
    ]

def extract_hash_emo(text, lexicon):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'sadness', 'surprise', 'trust']
    scores = {emo: [] for emo in emotions}

    for word in text.split():
        if word in lexicon:
            for emo, value in lexicon[word].items():
                if emo in scores:
                    scores[emo].append(value)

    return [np.mean(scores[emo]) if scores[emo] else 0.0 for emo in emotions]

# ensmeble
class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model1, model2, alpha=0.3):
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return self.alpha * pred1 + (1 - self.alpha) * pred2

# main
def main():
    # load datasset by calling the function abovee
    combined_df = load_datasets()
    if combined_df is None:
        st.stop()
    
    # laodf lexicons juga
    lexicons = load_lexicons()
    
    # define emotion columns
    emotion_cols = ["joy", "sadness", "anger", "fear"]
    
    # display daata
    st.subheader("📊 Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total samples:** {len(combined_df)}")
        st.write(f"**Test set size:** {test_size*100:.0f}%")
        st.write(f"**Columns:** {list(combined_df.columns)}")
        
    with col2:
        st.write("**🎯 Target Emotions:**")
        for col in emotion_cols:
            if col in combined_df.columns:
                st.write(f"- ✅ {col.title()}")
            else:
                st.write(f"- ❌ {col.title()} (missing)")
    
    missing_cols = [col for col in ['Tweet'] + emotion_cols if col not in combined_df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()
    
    # train model part
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models with your data... This may take several minutes."):
            
            # split data pake approach yg dimau
            train_df, test_df = train_test_split(combined_df, test_size=test_size, random_state=42)
            
            st.success(f"✅ Data split: Train ({len(train_df)}) | Test ({len(test_df)})")
            
            # apply text clean
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["clean_text"] = train_df["Tweet"].apply(clean_text)
            test_df["clean_text"] = test_df["Tweet"].apply(clean_text)
            
            # extract lexicon features
            # NRC Lexicon features
            train_df['lexicons'] = train_df['clean_text'].apply(lambda x: extract_lex(x, lexicons['nrc']))
            test_df['lexicons'] = test_df['clean_text'].apply(lambda x: extract_lex(x, lexicons['nrc']))
            train_lex = np.array(train_df['lexicons'].tolist())
            test_lex = np.array(test_df['lexicons'].tolist())
            
            # VAD features
            train_df['vad'] = train_df['clean_text'].apply(lambda x: extract_vad(x, lexicons['vad']))
            test_df['vad'] = test_df['clean_text'].apply(lambda x: extract_vad(x, lexicons['vad']))
            train_vad = np.array(train_df['vad'].tolist())
            test_vad = np.array(test_df['vad'].tolist())
            
            # HashEmo features
            train_df['hash'] = train_df['clean_text'].apply(lambda x: extract_hash_emo(x, lexicons['hash']))
            test_df['hash'] = test_df['clean_text'].apply(lambda x: extract_hash_emo(x, lexicons['hash']))
            train_hash = np.array(train_df['hash'].tolist())
            test_hash = np.array(test_df['hash'].tolist())
            
            # scale features
            scaler_hash = StandardScaler()
            train_hash = scaler_hash.fit_transform(train_hash)
            test_hash = scaler_hash.transform(test_hash)
            
            scaler_lex = StandardScaler()
            train_lex = scaler_lex.fit_transform(train_lex)
            test_lex = scaler_lex.transform(test_lex)
            
            scaler_vad = StandardScaler()
            train_vad = scaler_vad.fit_transform(train_vad)
            test_vad = scaler_vad.transform(test_vad)
            
            # combine theee features
            train_combined = np.concatenate([train_vad, train_lex, train_hash], axis=1)
            test_combined = np.concatenate([test_vad, test_lex, test_hash], axis=1)
            
            # tfidf vectorization
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            train_tfidf = tfidf.fit_transform(train_df['clean_text'])
            test_tfidf = tfidf.transform(test_df['clean_text'])
            
            # combien all features
            X_train = hstack([train_tfidf, train_combined])
            X_test = hstack([test_tfidf, test_combined])
            
            # prep target
            y_train_cls = train_df[emotion_cols].values.argmax(axis=1)
            y_test_cls = test_df[emotion_cols].values.argmax(axis=1)
            y_train_reg = train_df[emotion_cols].values
            y_test_reg = test_df[emotion_cols].values
            
            
            # train regress based on selection
            if selected_reg_model == "Ridge":
                ridge = Ridge(alpha=1.0, solver='lsqr', random_state=42)
                reg_model = MultiOutputRegressor(ridge)
                
            elif selected_reg_model == "LightGBM" and LIGHTGBM_AVAILABLE:
                lgbm_base = LGBMRegressor(
                    num_leaves=20,
                    n_estimators=500,
                    learning_rate=0.1,
                    reg_alpha=0.1,
                    min_child_samples=3,
                    colsample_bytree=0.3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                reg_model = MultiOutputRegressor(lgbm_base)
                
            elif "Ensemble" in selected_reg_model and LIGHTGBM_AVAILABLE:
                ridge = Ridge(alpha=1.0, solver='lsqr', random_state=42)
                ridge_reg = MultiOutputRegressor(ridge)
                
                lgbm_base = LGBMRegressor(
                    num_leaves=20,
                    n_estimators=500,
                    learning_rate=0.1,
                    reg_alpha=0.1,
                    min_child_samples=3,
                    colsample_bytree=0.3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                lgbm_reg = MultiOutputRegressor(lgbm_base)
                
                reg_model = EnsembleRegressor(model1=ridge_reg, model2=lgbm_reg, alpha=alpha)
            
            reg_model.fit(X_train, y_train_reg)
            y_pred_reg = reg_model.predict(X_test)
            
            # train classif by selection
            if selected_clf_model == "Logistic Regression":
                clf_model = LogisticRegression(max_iter=50, solver='newton-cg', random_state=42)
                
            elif selected_clf_model == "LightGBM" and LIGHTGBM_AVAILABLE:
                clf_model = LGBMClassifier(
                    num_leaves=20,
                    n_estimators=500,
                    earning_rate=0.1,
                    reg_alpha=0.1,
                    min_child_samples=3,
                    colsample_bytree=0.5,
                    random_state=42,
                    n_jobs=-1
                )
                
            elif selected_clf_model == "SVM":
                clf_model = SVC(kernel='linear', C=1.0, probability=True, max_iter=10000, random_state=42)
                
            elif "Ensemble" in selected_clf_model and LIGHTGBM_AVAILABLE:
                logR = LogisticRegression(max_iter=50, solver='newton-cg', random_state=42)
                lgbm_clf = LGBMClassifier(
                    num_leaves=20,
                    n_estimators=500,
                    learning_rate=0.1,
                    reg_alpha=0.1,
                    min_child_samples=3,
                    colsample_bytree=0.5,
                    random_state=42,
                    n_jobs=-1
                )
                clf_model = VotingClassifier(
                    estimators=[('lr', logR), ('lgbm', lgbm_clf)],
                    voting='soft'
                )
            
            clf_model.fit(X_train, y_train_cls)
            y_pred_cls = clf_model.predict(X_test)
            
            # store in session state
            st.session_state.reg_model = reg_model
            st.session_state.clf_model = clf_model
            st.session_state.tfidf = tfidf
            st.session_state.scaler_lex = scaler_lex
            st.session_state.scaler_vad = scaler_vad
            st.session_state.scaler_hash = scaler_hash
            st.session_state.lexicons = lexicons
            st.session_state.emotion_cols = emotion_cols
            st.session_state.models_trained = True
            
            # results
            st.success("✅ Models trained successfully!")
            
            # metrics regres
            st.subheader("📈 Regression Model Performance")
            reg_metrics = []
            for i, emotion in enumerate(emotion_cols):
                mae = mean_absolute_error(y_test_reg[:, i], y_pred_reg[:, i])
                mse = mean_squared_error(y_test_reg[:, i], y_pred_reg[:, i])
                r2 = r2_score(y_test_reg[:, i], y_pred_reg[:, i])
                corr, _ = pearsonr(y_test_reg[:, i], y_pred_reg[:, i])
                reg_metrics.append({
                    'Emotion': emotion.title(),
                    'MAE': f"{mae:.4f}",
                    'MSE': f"{mse:.4f}",
                    'R²': f"{r2:.4f}",
                    'Pearson': f"{corr:.4f}"
                })
            
            st.dataframe(pd.DataFrame(reg_metrics), use_container_width=True)
            
            # metrics classif
            st.subheader("🎯 Classification Model Performance")
            accuracy = accuracy_score(y_test_cls, y_pred_cls)
            st.metric("Overall Accuracy", f"{accuracy:.3f}")
            
            # classif report
            report = classification_report(y_test_cls, y_pred_cls, target_names=emotion_cols, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
    
    # emotion prediction interface
    st.markdown("---")
    st.subheader("🔮 Emotion Prediction")
    
    # text input value
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # text input with session state (session state basically allows the program to remember users data)
    user_text = st.text_area(
        "Enter text to analyze emotions:",
        value=st.session_state.input_text,
        placeholder="Type your emotional statement here... (e.g., 'I'm feeling really happy today!')",
        height=100,
        key="text_input"
    )
    
    # clear button d text box
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Text"):
            st.session_state.input_text = ""
            st.rerun()
    
    if st.button("🎭 Analyze Emotions") and user_text:
        if not hasattr(st.session_state, 'models_trained') or not st.session_state.models_trained:
            st.error("⚠️ Please train the models first!")
        else:
            with st.spinner("Analyzing emotions..."):
                # analyse panggil fungsi diatas
                results = EmoIntPipeline([user_text])
                text, emotion, intensity, confidence = results[0]
                
                # results display
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎭 Predicted Emotion", emotion.title())
                    st.metric("💪 Intensity Score", f"{intensity:.3f}")
                    st.metric("🎯 Confidence", f"{confidence:.3f}")
                
                with col2:
                    # show prediction
                    st.write("**Pipeline Output:**")
                    st.write(f"- **Text:** {text}")
                    st.write(f"- **Emotion:** {emotion}")
                    st.write(f"- **Intensity:** {intensity:.3f}")
                    st.write(f"- **Confidence:** {confidence:.3f}")
                
                # plotting buat visual rep
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # bar chart showing predicted emotions
                emotions = st.session_state.emotion_cols
                scores = [intensity if em == emotion else 0.1 for em in emotions]
                colors = ['red' if em == emotion else 'lightblue' for em in emotions]
                
                ax1.bar(emotions, scores, color=colors)
                ax1.set_title('Predicted Emotion Intensity')
                ax1.set_ylabel('Intensity')
                ax1.set_ylim(0, 1)
                plt.setp(ax1.get_xticklabels(), rotation=45)
                
                # confidence pie chart
                conf_data = [confidence, 1-confidence]
                ax2.pie(conf_data, labels=[f'{emotion.title()} ({confidence:.1%})', f'Other ({1-confidence:.1%})'], 
                       autopct='%1.1f%%', startangle=90,
                       colors=['red', 'lightgray'])
                ax2.set_title('Model Confidence')
                
                plt.tight_layout()
                st.pyplot(fig)

def EmoIntPipeline(texts):
    """Your original EmoIntPipeline function"""
    if isinstance(texts, str):
        texts = [texts]

    # models 
    reg_model = st.session_state.reg_model
    clf_model = st.session_state.clf_model
    tfidf = st.session_state.tfidf
    scaler_lex = st.session_state.scaler_lex
    scaler_vad = st.session_state.scaler_vad
    scaler_hash = st.session_state.scaler_hash
    lexicons = st.session_state.lexicons
    emotion_cols = st.session_state.emotion_cols

    #clean text and tfidf extraction
    texts = [clean_text(t) for t in texts]
    tfidf_feat = tfidf.transform(texts)

    # extract lexicons
    lex_feat = np.array([extract_lex(t, lexicons['nrc']) for t in texts])
    vad_feat = np.array([extract_vad(t, lexicons['vad']) for t in texts])
    hash_feat = np.array([extract_hash_emo(t, lexicons['hash']) for t in texts])

    # scale lexicons
    lex_feat = scaler_lex.transform(lex_feat)
    vad_feat = scaler_vad.transform(vad_feat)
    hash_feat = scaler_hash.transform(hash_feat)

    # combined lexicons
    combined_lex = np.concatenate([vad_feat, lex_feat, hash_feat], axis=1)
    combined_feat = hstack([tfidf_feat, combined_lex])

    # classif
    probs = clf_model.predict_proba(combined_feat)
    confidence_scores = np.max(probs, axis=1)
    predicted_classes = np.argmax(probs, axis=1)
    pred_emotion = [emotion_cols[i] for i in predicted_classes]

    # regres
    pred_reg_all = reg_model.predict(combined_feat)
    pred_intensity = [pred[i] for pred, i in zip(pred_reg_all, predicted_classes)]

    rets = []
    for text, em, score, conf in zip(texts, pred_emotion, pred_intensity, confidence_scores):
        if conf > 0.4 and score < 0.4:
            boost = 0.3 * conf
            score = min(score + boost, 1.0)
        rets.append((text, em, score, conf))

    return rets

def show_sample_texts():
    """Show sample texts for quick testing"""
    st.markdown("---")
    st.subheader("🔬 Quick Test Samples")
    
    # original sample texts yang bisa diopakai user juga
    sample_texts = [
        "I'm so happy and grateful today!",
        "I'm kinda happy today, but at the same time life is so bland",
        "I might be happy today, but it's just a normal day.",
        "This makes me so angry, I can't believe they did that.",
        "I'm angry, but I think I can tolerate their behavior.",
        "I'm extremely furious, he never get things right.",
        "I'm feeling a bit down today, things aren't going as planned.",
        "My girlfriend just dumped me, I don't know what to do with my life anymore.",
        "I'm crying my eyes out now, a family member of mine just passed away.",
        "That movie was terrifying, I couldn't sleep all night.",
        "The haunted house was scary, but we had so much fun",
        "That scared me so much, I almost had a heart attack."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"Sample {i}: {text}")
        with col2:
            if st.button(f"Use This", key=f"sample_{i}"):
                # update session state biar bisa input text ke box inputtan
                st.session_state.input_text = text
                st.success(f"✅ Text copied to input box!")
                st.rerun()

    # pas test di klik di yg sample texts
    if hasattr(st.session_state, 'input_text') and st.session_state.input_text:
        st.info("💡 Sample text has been copied to the input box above. You can now analyze it!")


if __name__ == "__main__":
    main()
    show_sample_texts()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎭 Emotion Detection & Intensity Analysis Tool</p>
        <p><small>Kelompok 7 - Prediction</small></p>
    </div>
    """, unsafe_allow_html=True)