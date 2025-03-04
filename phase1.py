from google import genai
import pandas as pd
import time

# Load dataset
data = pd.read_csv("Dataset_CultureVLM_style - Concepts Introductions.csv")

# Initialize Google genai Client
client = genai.Client(api_key="AIzaSyCF7xCRjmobiB15TrYAuqh4tYrOxx4vWMs")
# Prepare dataframe for storing results
introductions_df = pd.DataFrame()
introductions_df["concept"] = data["concept"]


# Function to generate introductions
def get_introduction(concept):
    prompt = f"""
    Concept: {concept}
    
    Please provide a detailed introduction of {concept} of Egypt, including:
    - Location and Features (where it is found or originates from, what makes it unique).
    - Time period (when it was created or became significant).
    - History (historical background and development, or any significant events).
    - Cultural significance (cultural context in Egypt, modern-day significance).
    - Stories or Legends (any stories, legends, or folklore associated with {concept}).

    Use the following format:
    Introduction of {concept}: [Detailed Introduction of {concept}]
    """

    try:
        # Generate content using Google's genai
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Ensure the model is correctly set
            contents=prompt,
        )

        introduction = (
            response.text.strip()
            if response and hasattr(response, "text")
            else "No introduction found."
        )
        print(f"Generated introduction for {concept}:\n{introduction}\n")
        return introduction

    except Exception as e:
        print(f"Error generating introduction for {concept}: {e}")
        return ""  # Return an empty string in case of an error


# Process each concept
introductions = []
for concept in data["concept"]:
    intro = get_introduction(concept)
    introductions.append(intro)
    time.sleep(10)  # Add delay to avoid hitting rate limits

# Store results in DataFrame and save to CSV
introductions_df["introduction"] = pd.Series(introductions)
introductions_df.to_csv("concept_introductions.csv", index=False)

print("Introductions saved to concept_introductions.csv")
