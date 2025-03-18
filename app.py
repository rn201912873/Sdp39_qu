import streamlit as st
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from db import (get_database, hash_password, verify_password,add_chat_history, get_chat_history, clear_chat_history)
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
import time

# Initialize MongoDB connection
db = get_database()
users_collection = db['users']
chat_history_collection = db['chat_history'] 

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'signup' not in st.session_state:
    st.session_state['signup'] = False
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'chat_title' not in st.session_state:
    st.session_state['chat_title'] = "New Chat"
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

# User authentication

# Define class_labels at the module level
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Update export function to ensure class_labels is available
def export_chat_history_to_pdf(history, username, session_id=None):
    global class_labels  # Make sure we can access the global class_labels
    
    if not history:
        raise ValueError("No chat history to export")
    
    # Create exports directory if it doesn't exist
    export_dir = "d:/QU/sdp-39-cs-m/exports"
    os.makedirs(export_dir, exist_ok=True)
        
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Filter by session_id if provided
    if session_id:
        history = [chat for chat in history if chat.get('session_id') == session_id]
    
    # Group chats by session
    sessions = {}
    for chat in history:
        chat_session_id = chat.get('session_id', chat['timestamp'].split()[0])
        if chat_session_id not in sessions:
            sessions[chat_session_id] = []
        sessions[chat_session_id].append(chat)
    
    for sess_id, chats in sessions.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"Session: {sess_id}", ln=True, align='C')
        
        # Add session title if exists
        session_title = chats[0].get('chat_title', 'Untitled Session')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Title: {session_title}", ln=True)
        
        for chat in sorted(chats, key=lambda x: x['timestamp']):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"Time: {chat['timestamp']}", ln=True)
            pdf.cell(200, 10, txt=f"Image: {chat['image']}", ln=True)
            pdf.cell(200, 10, txt=f"Model: {chat['model_name']}", ln=True)
            
            if 'comparison_results' in chat:
                pdf.cell(200, 10, txt="Comparison Results:", ln=True)
                # Save comparison chart using matplotlib
                fig = display_comparison_chart_for_pdf([{
                    "name": r["model"],
                    "probabilities": [r["probabilities"]], 
                    "predicted_class_idx": class_labels.index(r["predicted_class"])
                } for r in chat['comparison_results']], class_labels)
                
                chart_path = f"temp_comparison_{chat['timestamp'].replace(':', '-')}.png"
                fig.savefig(chart_path, bbox_inches='tight')
                pdf.image(chart_path, x=10, w=190)
                os.remove(chart_path)
                
                for result in chat['comparison_results']:
                    pdf.cell(200, 10, txt=f"Model: {result['model']}", ln=True)
                    pdf.cell(200, 10, txt=f"Prediction: {result['predicted_class']}", ln=True)
                    pdf.cell(200, 10, txt=f"Confidence: {result['confidence']*100:.1f}%", ln=True)
            else:
                # Save single prediction chart using matplotlib
                fig = create_prediction_chart_for_pdf([chat['probabilities']], class_labels)
                chart_path = f"temp_single_{chat['timestamp'].replace(':', '-')}.png"
                fig.savefig(chart_path, bbox_inches='tight')
                pdf.cell(200, 10, txt=f"Prediction: {chat['predicted_class']}", ln=True)
                pdf.cell(200, 10, txt=f"Confidence: {chat['confidence']*100:.1f}%", ln=True)
                pdf.image(chart_path, x=10, w=190)
                os.remove(chart_path)
            
            if chat.get('attack_type') and chat['attack_type'] != "None":
                pdf.cell(200, 10, txt=f"Attack: {chat['attack_type']}", ln=True)
    
    # Save PDF with appropriate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"brain_tumor_report_{username}_{timestamp}.pdf"
    if session_id:
        filename = f"brain_tumor_report_{username}_{session_id}_{timestamp}.pdf"
    
    filepath = os.path.join("d:/QU/sdp-39-cs-m/exports", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath)
    return filepath

def create_prediction_chart_for_pdf(probabilities, class_labels, title="Prediction Probabilities"):
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, [float(p) * 100 for p in probabilities[0]])
    plt.title(title)
    plt.ylabel('Confidence (%)')
    plt.xlabel('Class')
    plt.ylim(0, 100)
    
    # Add value labels on top of each bar
    for i, v in enumerate([float(p) * 100 for p in probabilities[0]]):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    return plt

def display_comparison_chart_for_pdf(models_data, class_labels):
    plt.figure(figsize=(12, 6))
    bar_width = 0.8 / len(models_data)
    
    for idx, model_data in enumerate(models_data):
        x = np.arange(len(class_labels)) + idx * bar_width
        values = [float(p) * 100 for p in model_data["probabilities"][0]]
        plt.bar(x, values, bar_width, label=model_data["name"])
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(x[i], v + 1, f'{v:.1f}%', ha='center', va='bottom', rotation=90)
    
    plt.xlabel('Class')
    plt.ylabel('Confidence (%)')
    plt.title('Model Comparison')
    plt.xticks(np.arange(len(class_labels)) + bar_width * (len(models_data)-1)/2, class_labels)
    plt.legend()
    plt.ylim(0, 100)
    
    return plt

def create_prediction_chart(probabilities, class_labels, title="Prediction Probabilities"):
    fig = go.Figure(data=[
        go.Bar(
            x=class_labels,
            y=[float(p) * 100 for p in probabilities[0]],
            text=[f'{float(p)*100:.1f}%' for p in probabilities[0]],
            textposition='auto',
            marker_color='rgb(26, 118, 255)'
        )
    ])
    
    fig.update_layout(
        title=title,
        yaxis_title='Confidence (%)',
        xaxis_title='Class',
        yaxis_range=[0,100],
        template='plotly_white'
    )
    
    return fig

def display_comparison_chart(models_data, class_labels):
    fig = go.Figure()
    for model_data in models_data:
        fig.add_trace(go.Bar(
            name=model_data["name"],
            x=class_labels,
            y=[float(p) * 100 for p in model_data["probabilities"][0]],
            text=[f'{float(p)*100:.1f}%' for p in model_data["probabilities"][0]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Comparison",
        yaxis_title='Confidence (%)',
        xaxis_title='Class',
        yaxis_range=[0,100],
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def login():
    st.title("Brain Tumor Classification Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = users_collection.find_one({"username": username})
        if user and verify_password(password, user["password"]):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

def signup():
    st.title("Brain Tumor Classification Signup")
    if st.button("← Back to Login"):
        st.session_state.signup = False
        st.rerun()
    
    with st.form("signup_form"):
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            # Validate empty fields
            if not username or not password or not confirm_password:
                st.error("Please fill in all fields")
                return
            
            # Validate username length
            if len(username) < 3:
                st.error("Username must be at least 3 characters long")
                return
            
            # Validate username characters
            if not username.isalnum():
                st.error("Username must contain only letters and numbers")
                return
                
            # Check if username exists
            if users_collection.find_one({"username": username}):
                st.error("Username already exists")
                return
                
            # Password validation
            if len(password) < 6:
                st.error("Password must be at least 6 characters long")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            # Create new user
            hashed_password = hash_password(password)
            users_collection.insert_one({
                "username": username,
                "password": hashed_password
            })
            st.success("Signup successful! Please login.")
            st.session_state.signup = False
            st.rerun()

def logout():
    st.session_state.logged_in = False

if 'signup' not in st.session_state:
    st.session_state.signup = False

if not st.session_state.logged_in:
    if st.session_state.signup:
        signup()
    else:
        login()
        if st.button("Need an account? Signup"):
            st.session_state.signup = True
    st.stop()
else:
    st.sidebar.button("Logout", on_click=logout)

    # Create tabs for main content
    tab1, tab2 = st.tabs(["Classification", "Chat History"])
    
    with tab1:
        st.title("Brain Tumor Classification")

        # Interaction history
        if 'history' not in st.session_state:
            st.session_state.history = get_chat_history(db, st.session_state.username)

        # Add export button for current session
        if st.session_state.history:
            current_session_chats = [
                chat for chat in st.session_state.history 
                if chat.get('session_id') == st.session_state.session_id
            ]
            if current_session_chats:
                if st.button("📥 Export Current Session"):
                    try:
                        filepath = export_chat_history_to_pdf(
                            current_session_chats,
                            st.session_state.username,
                            st.session_state.session_id
                        )
                        st.success(f"Session exported to {filepath}")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")

        # Update model options to include all combinations
        model_options = [
            "Clean Model", 
            "Exposure Model",
            "Distortion Model",
            "Combined Model",
            "Compare Clean vs Exposure",
            "Compare Clean vs Distortion",
            "Compare Clean vs Combined",
            "Compare Exposure vs Distortion",
            "Compare Exposure vs Combined",
            "Compare Distortion vs Combined",
            "Compare All Models"
        ]
        selected_model = st.sidebar.selectbox("Select Model", model_options)

        # Update attack options to include exposure
        attack_options = ["None", "Gaussian Noise", "Exposure"]
        selected_attack = st.sidebar.selectbox("Select Attack", attack_options)

        # Function to add Gaussian noise to the image
        def add_gaussian_noise(image, mean=0.0, std=0.1):
            noise = torch.randn(image.size(), device=image.device) * std + mean
            noisy_image = image + noise
            noisy_image = torch.clamp(noisy_image, 0., 1.)  # Clip values to maintain valid image range
            return noisy_image

        # Add exposure adjustment function
        def adjust_exposure(image, exposure_factor_range=(0.5, 1.5), apply_prob=0.5):
            if np.random.random() < apply_prob:
                exposure_factor = np.random.uniform(*exposure_factor_range)
                image = image * exposure_factor
                image = torch.clamp(image, 0, 1)
            return image

        # Update model loading function
        @st.cache_resource  # Cache the model to avoid reloading on every interaction
        def load_model(model_name):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"Using device: {device}")

            model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small", num_labels=4, ignore_mismatched_sizes=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

            # Map both full names and short names to model paths
            base_models = {
                "Clean": r"D:\QU\sdp-39-cs-m\models\fine_tuned_brain_tumor_model.pth",
                "Exposure": r"D:\QU\sdp-39-cs-m\models\fine_tuned_brain_tumor_model_with_exposure.pth",
                "Distortion": r"D:\QU\sdp-39-cs-m\models\fine_tuned_brain_tumor_model_with_distortion.pth",
                "Combined": r"D:\QU\sdp-39-cs-m\models\fine_tuned_brain_tumor_model_with_noise_and_exposure_combined.pth"
            }
            
            model_path_map = {**base_models}  # Copy base models
            # Add full names
            for key in base_models:
                model_path_map[f"{key} Model"] = base_models[key]

            # Add error handling for model path lookup
            try:
                model_path = model_path_map[model_name]
            except KeyError:
                st.error(f"Invalid model name: {model_name}")
                st.error("Available models: " + ", ".join(model_path_map.keys()))
                raise

            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)

            model = model.to(device)
            model.eval()

            image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
            return model, image_processor, device

        # Upload image
        uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            if st.button("Classify"):
                # Initialize variables for comparison results
                comparison_data = None
                models_data = []
                
                with st.spinner("Classifying..."):
                    if selected_model in ["Clean Model", "Exposure Model", "Distortion Model", "Combined Model"]:
                        model, image_processor, device = load_model(selected_model)
                        inputs = image_processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        # Apply attack if selected
                        if selected_attack == "Gaussian Noise":
                            inputs["pixel_values"] = add_gaussian_noise(inputs["pixel_values"])
                        elif selected_attack == "Exposure":
                            inputs["pixel_values"] = adjust_exposure(inputs["pixel_values"])

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probabilities = torch.softmax(logits, dim=-1)
                            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

                        probabilities = probabilities.cpu()

                        st.subheader("Classification Results")
                        st.write(f"**Predicted Class**: {class_labels[predicted_class_idx]}")
                        st.write(f"**Confidence**: {probabilities[0][predicted_class_idx] * 100:.1f}%")
                        
                        # Display interactive chart
                        fig = create_prediction_chart(probabilities, class_labels)
                        st.plotly_chart(fig, use_container_width=True, key=f"single_classification_{st.session_state.session_id}_{int(time.time())}")

                        # Save interaction history with timestamp
                        interaction = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "session_id": st.session_state.session_id,
                            "image": uploaded_file.name,
                            "predicted_class": class_labels[predicted_class_idx],
                            "confidence": float(probabilities[0][predicted_class_idx].item()),
                            "probabilities": [float(p) for p in probabilities[0].tolist()],
                            "model_name": selected_model,
                            "attack_type": selected_attack,
                            "chat_title": st.session_state.chat_title  # Add chat_title to interaction
                        }
                        
                        # Save to MongoDB and update session state
                        add_chat_history(db, st.session_state.username, interaction)
                        st.session_state.history = get_chat_history(db, st.session_state.username)

                    elif selected_model.startswith("Compare"):
                        is_all_models = selected_model == "Compare All Models"
                        model_names = (
                            ["Clean Model", "Exposure Model", "Distortion Model", "Combined Model"]
                            if is_all_models
                            else selected_model.replace("Compare ", "").split(" vs ")
                        )
                        
                        for model_name in model_names:
                            model, image_processor, device = load_model(model_name.strip())
                            # Process image and get predictions
                            inputs = image_processor(images=image, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            
                            if selected_attack == "Gaussian Noise":
                                inputs["pixel_values"] = add_gaussian_noise(inputs["pixel_values"])
                            elif selected_attack == "Exposure":
                                inputs["pixel_values"] = adjust_exposure(inputs["pixel_values"])
                            
                            with torch.no_grad():
                                outputs = model(**inputs)
                                logits = outputs.logits
                                probabilities = torch.softmax(logits, dim=-1)
                                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                                
                            models_data.append({
                                "name": model_name.strip(),
                                "probabilities": probabilities.cpu(),
                                "predicted_class_idx": predicted_class_idx
                            })
                        
                        # Display results and charts
                        for model_data in models_data:
                            st.subheader(f"Classification Results ({model_data['name']})")
                            st.write(f"**Predicted Class**: {class_labels[model_data['predicted_class_idx']]}")
                            st.write(f"**Confidence**: {model_data['probabilities'][0][model_data['predicted_class_idx']] * 100:.1f}%")
                            
                            st.write(f"**Probabilities for All Classes ({model_data['name']}):**")
                            for i, label in enumerate(class_labels):
                                st.write(f"{label}:")
                                st.progress(int(model_data['probabilities'][0][i] * 100))

                        # Display comparison chart
                        fig = display_comparison_chart(models_data, class_labels)
                        st.plotly_chart(fig, use_container_width=True, key=f"compare_classification_{st.session_state.session_id}_{int(time.time())}")

                        # Create comparison data structure
                        comparison_data = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "session_id": st.session_state.session_id,
                            "image": uploaded_file.name,
                            "model_name": selected_model,
                            "attack_type": selected_attack,
                            "is_comparison": True,
                            "comparison_results": [{
                                "model": data["name"],
                                "predicted_class": class_labels[data["predicted_class_idx"]],
                                "confidence": float(data["probabilities"][0][data["predicted_class_idx"]].item()),
                                "probabilities": [float(p) for p in data["probabilities"][0].tolist()]
                            } for data in models_data],
                            "chat_title": st.session_state.chat_title  
                        }
                        
                        # Save comparison results
                        if comparison_data:
                            add_chat_history(db, st.session_state.username, comparison_data)
                            st.session_state.history = get_chat_history(db, st.session_state.username)

    with tab2:
        st.title("Chat History")
        
        # Create a container for state management
        state_container = st.container()
        
        with state_container:
            # Display all chats grouped by session
            chats = get_chat_history(db, st.session_state.username)
            if chats:
                # Group chats by session
                sessions = {}
                for chat in chats:
                    session_id = chat.get('session_id', chat['timestamp'].split()[0])
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(chat)
                
                # Sort sessions by date (newest first)
                sorted_sessions = dict(sorted(sessions.items(), key=lambda x: x[0], reverse=True))
                
                for session_id, session_chats in sorted_sessions.items():
                    with st.expander(f"Session: {session_id}", expanded=True):
                        # Display chat entries
                        for i, chat in enumerate(sorted(session_chats, key=lambda x: x['timestamp'])):
                            with st.container():
                                st.write("---")
                                st.write(f"**Time**: {chat['timestamp']}")
                                st.write(f"**Image**: {chat['image']}")
                                st.write(f"**Model**: {chat['model_name']}")
                                
                                if chat.get('is_comparison', False):
                                    st.write("**Comparison Results:**")
                                    # for result in chat['comparison_results']:
                                    #     st.write(f"*{result['model']}*")
                                    #     st.write(f"Predicted: ...")
                                    #     st.write(f"Confidence: ...")

                                    # Keep only the combined comparison chart:
                                    fig = display_comparison_chart([{
                                        "name": r["model"],
                                        "probabilities": [r["probabilities"]],
                                        "predicted_class_idx": class_labels.index(r["predicted_class"])
                                    } for r in chat['comparison_results']], class_labels)
                                    st.plotly_chart(fig, use_container_width=True, key=f"historical_comparison_{session_id}_{chat['timestamp']}")
                                else:
                                    # Regular single model results display
                                    try:
                                        st.write(f"**Predicted Class**: {chat['predicted_class']}")
                                        st.write(f"**Confidence**: {chat['confidence']*100:.1f}%")
                                        
                                        fig = create_prediction_chart(
                                            [chat['probabilities']], 
                                            class_labels,
                                            f"Prediction Results"
                                        )
                                        st.plotly_chart(fig, use_container_width=True, 
                                                      key=f"chart_{session_id}_{chat['timestamp']}")
                                    except KeyError as e:
                                        st.error(f"Error displaying results: Missing {str(e)}")
                                
                                if chat.get('attack_type') and chat['attack_type'] != "None":
                                    st.write(f"**Attack Type**: {chat['attack_type']}")

        # Export button
        if st.button("Export Chat History"):
            try:
                filepath = export_chat_history_to_pdf(chats, st.session_state.username)
                st.success(f"Chat history exported successfully to {filepath}")
            except Exception as e:
                st.error(f"Error exporting chat history: {str(e)}")

    # Move the About section to the sidebar
    st.sidebar.subheader("About")
    st.sidebar.write("""
    This application uses a fine-tuned MobileViT model for brain tumor classification.
    The vision is to leverage AI to assist in medical diagnostics, providing accurate and
    efficient analysis of MRI images to detect brain tumors.
    """)

    # Update comparison visualization
    def display_comparison_chart(models_data, class_labels):
        fig = go.Figure()
        for model_data in models_data:
            fig.add_trace(go.Bar(
                name=model_data["name"],
                x=class_labels,
                y=[float(p) * 100 for p in model_data["probabilities"][0]],
                text=[f'{float(p)*100:.1f}%' for p in model_data["probabilities"][0]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Comparison",
            yaxis_title='Confidence (%)',
            xaxis_title='Class',
            yaxis_range=[0,100],
            barmode='group',
            template='plotly_white'
        )
        
        return fig






