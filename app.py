import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
import os
import time

# --- Configuration ---
# Must match the constants used during training
MAX_LEN = 50
MODEL_PATH = 'sentiment_analysis_model.h5'
TOKENIZER_PATH = "tokenizer (1).pickle"

# Initialize session state for history and navigation
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "New Analysis"
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# --- Helper Functions ---

# Function to load model and tokenizer
@st.cache_resource
def load_assets():
    """Loads the model and tokenizer, providing feedback if files are missing."""
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load the Tokenizer
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except FileNotFoundError:
        st.error(f"Required files not found. Please ensure both '{MODEL_PATH}' and '{TOKENIZER_PATH}' exist. Run 'train_and_save.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during asset loading: {e}")
        st.stop()

# Function to preprocess text and make prediction
def predict_sentiment(text, model, tokenizer):
    """Tokenizes, pads, and predicts sentiment for the given text."""
    # 1. Tokenize
    sequence = tokenizer.texts_to_sequences([text])
    
    # 2. Pad
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    # Sigmoid output is the probability of class 1 (Positive)
    prob_positive = float(prediction)
    prob_negative = 1.0 - prob_positive
    
    return prob_negative, prob_positive

# --- Streamlit App UI and Logic ---

# Set a beautiful page configuration (Tailwind aesthetics)
st.set_page_config(
    page_title="Bi-LSTM Sentiment Analyzer",
    layout="wide",  # Changed to wide for better sidebar layout
    initial_sidebar_state="expanded",  # Start with sidebar open
)

# Custom Styling (using Markdown for simple CSS injection)
st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fb;
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333333;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-box textarea {
        border: 10px solid #e0e0e0;
        border-radius: 30px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .input-box textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .result-container {
        margin-top: 2rem;
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
    }
    .positive-border { border-left-color: #10b981; }
    .negative-border { border-left-color: #ef4444; }
    /* Sidebar styling */
    .sidebar-section {
        padding: 10px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .history-item {
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 8px;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .history-item:hover {
        background-color: #e9ecef;
    }
    .history-text {
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-sentiment {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Load assets (Model and Tokenizer)
model, tokenizer = load_assets()

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧠 Sentiment Analyzer")
    
    # Navigation menu
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    selected_page = st.radio(
        "Navigation",
        ["🏠 New Analysis"],
        key="navigation"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # New Chat button
    if st.button("➕ New Analysis", use_container_width=True):
        st.session_state.current_page = "New Analysis"
        st.session_state.current_result = None
        st.rerun()
    
    # Clear History button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared!")
        time.sleep(1)
        st.rerun()
    
    # Recent Reviews History
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📜 Recent Reviews")
    
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-10:])):  # Show last 10 items
            sentiment_class = "positive" if item['sentiment'] == "Positive" else "negative"
            if st.button(f"{item['text'][:50]}...", key=f"history_{idx}", help=f"Sentiment: {item['sentiment']} ({item['confidence']:.1f}%)"):
                st.session_state.current_result = item
                st.session_state.current_page = "Review Detail"
                st.rerun()
            
            # Display sentiment and confidence
            st.markdown(f'<div class="history-sentiment {sentiment_class}">{item["sentiment"]} ({item["confidence"]:.1f}%)</div>', unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No reviews yet. Start analyzing!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Stats
    if st.session_state.history:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📊 Quick Stats")
        total_reviews = len(st.session_state.history)
        positive_count = sum(1 for item in st.session_state.history if item['sentiment'] == "Positive")
        negative_count = total_reviews - positive_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_reviews)
        with col2:
            st.metric("Positive %", f"{(positive_count/total_reviews*100):.1f}%" if total_reviews > 0 else "0%")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content Based on Navigation ---

if selected_page == "🏠 New Analysis" or st.session_state.current_page == "New Analysis":
    # --- Title and Description ---
    st.markdown('<div class="main-title">Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting Positive (1) or Negative (0) sentiment using a Bi-LSTM network.</div>', unsafe_allow_html=True)

    # --- User Input ---
    user_input = st.text_area(
        "Enter a piece of text (e.g., a tweet or review):",
        placeholder="Example: I am extremely happy with my new phone!",
        key="input_text",
        height=150
    )
    st.markdown('<div class="input-box"></div>', unsafe_allow_html=True)

    # --- Prediction Button ---
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Show a loading spinner while predicting
            with st.spinner('Analyzing...'):
                time.sleep(0.5) # Simulate a small delay for better user experience
                
                # Get prediction probabilities
                prob_neg, prob_pos = predict_sentiment(user_input, model, tokenizer)
                
                # Determine the final sentiment result
                if prob_pos > 0.5:
                    sentiment_result = "Positive"
                    main_prob = prob_pos
                    bar_class = "positive-border"
                else:
                    sentiment_result = "Negative"
                    main_prob = prob_neg
                    bar_class = "negative-border"
                
                # Save to history
                history_item = {
                    'text': user_input,
                    'sentiment': sentiment_result,
                    'confidence': main_prob * 100,
                    'prob_neg': prob_neg,
                    'prob_pos': prob_pos,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.history.append(history_item)
                
                # --- Display Results ---
                
                # Result Card
                st.markdown(f'<div class="result-container {bar_class}">', unsafe_allow_html=True)
                st.subheader(f"Predicted Sentiment: {sentiment_result}")
                
                # Custom confidence display with background color based on sentiment
                if sentiment_result == "Negative":
                    st.markdown(f'<div style="background-color: #fee2e2; padding: 10px; border-radius: 5px; border-left: 4px solid #ef4444;"><strong style="color: #dc2626;">Confidence: {main_prob * 100:.2f}%</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color: #dcfce7; padding: 10px; border-radius: 5px; border-left: 4px solid #10b981;"><strong style="color: #16a34a;">Confidence: {main_prob * 100:.2f}%</strong></div>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

                
                # --- Probability Plot ---
                st.markdown("---")
                st.subheader("Probability Distribution")
                
                # Create a DataFrame for the bar chart
                chart_data = pd.DataFrame({
                    'Class': ['Negative (0)', 'Positive (1)'],
                    'Probability': [prob_neg, prob_pos]
                })
                
                # Use Streamlit's built-in bar chart with custom colors
                # Create a bar chart with custom color mapping
                import altair as alt
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x='Class',
                    y='Probability',
                    color=alt.Color('Class', scale=alt.Scale(
                        domain=['Negative (0)', 'Positive (1)'],
                        range=['#ef4444', '#10b981']  # Custom colors: red for negative, green for positive
                    ))
                ).properties(
                    width=400,
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Optional detailed probabilities with custom colors based on sentiment
                col1, col2 = st.columns(2)
                
                if sentiment_result == "Negative":
                    # For negative sentiment: red color for negative probability, green for positive
                    col1.markdown(f'<div style="color: #ef4444; font-weight: bold;">Negative Probability: {prob_neg * 100:.2f}%</div>', unsafe_allow_html=True)
                    col2.markdown(f'<div style="color: #10b981; font-weight: bold;">Positive Probability: {prob_pos * 100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    # For positive sentiment: red color for negative probability, green for positive
                    col1.markdown(f'<div style="color: #ef4444; font-weight: bold;">Negative Probability: {prob_neg * 100:.2f}%</div>', unsafe_allow_html=True)
                    col2.markdown(f'<div style="color: #10b981; font-weight: bold;">Positive Probability: {prob_pos * 100:.2f}%</div>', unsafe_allow_html=True)

elif selected_page == "📊 Dashboard":
    st.markdown('<div class="main-title">Analysis Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # Overall statistics
        total_reviews = len(st.session_state.history)
        positive_count = sum(1 for item in st.session_state.history if item['sentiment'] == "Positive")
        negative_count = total_reviews - positive_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", total_reviews)
        with col2:
            st.metric("Positive Reviews", positive_count)
        with col3:
            st.metric("Negative Reviews", negative_count)
        
        # Sentiment distribution chart
        st.markdown("---")
        st.subheader("Sentiment Distribution")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [positive_count, negative_count]
        })
        
        chart = alt.Chart(sentiment_data).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color=alt.Color('Sentiment', scale=alt.Scale(
                domain=['Positive', 'Negative'],
                range=['#10b981', '#ef4444']
            ))
        ).properties(width=400, height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Recent reviews table
        st.markdown("---")
        st.subheader("Recent Reviews")
        
        recent_df = pd.DataFrame(st.session_state.history[-20:])  # Last 20 reviews
        if not recent_df.empty:
            st.dataframe(recent_df[['text', 'sentiment', 'confidence', 'timestamp']], use_container_width=True)
        
    else:
        st.info("No reviews yet. Start analyzing to see your dashboard!")

elif selected_page == "📈 Statistics":
    st.markdown('<div class="main-title">Detailed Statistics</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # Advanced statistics
        df = pd.DataFrame(st.session_state.history)
        
        # Average confidence by sentiment
        st.subheader("Average Confidence by Sentiment")
        avg_confidence = df.groupby('sentiment')['confidence'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Positive' in avg_confidence:
                st.metric("Positive Reviews Avg Confidence", f"{avg_confidence['Positive']:.2f}%")
        with col2:
            if 'Negative' in avg_confidence:
                st.metric("Negative Reviews Avg Confidence", f"{avg_confidence['Negative']:.2f}%")
        
        # Confidence distribution
        st.markdown("---")
        st.subheader("Confidence Distribution")
        
        confidence_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('confidence:Q', bin=True, title="Confidence (%)"),
            y='count()',
            color='sentiment'
        ).properties(width=600, height=400)
        
        st.altair_chart(confidence_chart, use_container_width=True)
        
        # Time-based analysis (if timestamps are available)
        st.markdown("---")
        st.subheader("Analysis Over Time")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            daily_counts = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            time_chart = alt.Chart(daily_counts).mark_line().encode(
                x='date:T',
                y='count:Q',
                color='sentiment:N'
            ).properties(width=600, height=400)
            
            st.altair_chart(time_chart, use_container_width=True)
        
    else:
        st.info("No reviews yet. Start analyzing to see statistics!")

elif selected_page == "⚙️ Settings":
    st.markdown('<div class="main-title">Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analysis Settings")
        
        # Model confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=50,
            max_value=95,
            value=50,
            help="Minimum confidence required to display results"
        )
        
        # Maximum history items
        max_history = st.number_input(
            "Maximum History Items",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum number of reviews to keep in history"
        )
        
        if st.button("Apply Settings"):
            st.session_state.confidence_threshold = confidence_threshold
            st.session_state.max_history = max_history
            st.success("Settings applied!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.subheader("Data Management")
        
        if st.button("Export History"):
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="sentiment_analysis_history.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No history to export!")
        
        if st.button("Clear All Data"):
            if st.checkbox("I confirm I want to clear all data"):
                st.session_state.history = []
                st.success("All data cleared!")
                time.sleep(1)
                st.rerun()

# Handle review detail view from history
if st.session_state.current_page == "Review Detail" and st.session_state.current_result:
    result = st.session_state.current_result
    st.markdown('<div class="main-title">Review Details</div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to New Analysis"):
        st.session_state.current_page = "New Analysis"
        st.session_state.current_result = None
        st.rerun()
    
    # Display the review details
    st.markdown("---")
    st.subheader("Review Text")
    st.write(result['text'])
    
    st.subheader("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Sentiment", result['sentiment'])
    with col2:
        st.metric("Confidence", f"{result['confidence']:.2f}%")
    
    st.subheader("Probability Distribution")
    prob_data = pd.DataFrame({
        'Class': ['Negative (0)', 'Positive (1)'],
        'Probability': [result['prob_neg'], result['prob_pos']]
    })
    
    chart = alt.Chart(prob_data).mark_bar().encode(
        x='Class',
        y='Probability',
        color=alt.Color('Class', scale=alt.Scale(
            domain=['Negative (0)', 'Positive (1)'],
            range=['#ef4444', '#10b981']
        ))
    ).properties(width=400, height=300)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.caption(f"Analyzed on: {result['timestamp']}")
