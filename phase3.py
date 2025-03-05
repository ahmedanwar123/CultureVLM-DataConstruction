from google import genai
import pandas as pd
import re
import time

# Load data
data = pd.read_csv("Dataset_CultureVLM_style - Concepts Introductions.csv")
questions_phase1 = pd.DataFrame()
questions_phase1["concept"] = data["concept"]

# Initialize Google genai Client
client = genai.Client(api_key="")

pattern = r"</think>([\s\S]*)"
questions = []

for index, row in data.iterrows():
    concept = row["concept"]
    introduction = row["intro"] if pd.notna(row["intro"]) else ""
    country = "Egypt"

    # Define the new structured prompt
    prompt = f"""Assume you are given a public image that shows the \"{concept}\" of {country}. Generate a visual reasoning \
    multiple-choice question based on this image and the introduction of {concept}.
    Provide the correct answer and reason immediately following the question.
    Here are some requirements:
    - The question must describe a specific scenario crafted to test deeper cultural
      understanding without directly naming {concept}. The scenario can be related to
      cultural background, regional characteristics, historical legends, or etiquette and
      customs, etc.
    - The question needs to be related to the image but does not need to describe the
      content of the image.
    - Ensure the question requires the user to recognize the image and use relevant
      knowledge to answer through reasoning based on the scenario provided. Users can only
      answer based on the image, so don’t mention any \"introduction\" or \"{concept}\" in the
      question.
    
    Use the following format:
    - Question: [Your Scenario-based Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D) [Option 4]
    - Answer: (X) [Option X]
    - Reason: [Your Reason for the Answer]
    Here are two examples:
    Image: Peking Duck
    Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in
    Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)
    and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is
    known for its thin, crispy skin and is served with pancakes, hoisin sauce, and
    scallions. The preparation involves inflating the duck to separate the skin from the
    fat, marinating it, and roasting it in a closed or hung oven. This meticulous process
    ensures the skin becomes crispy while the meat remains tender. Peking Duck is often
    carved in front of diners and served in three stages: the skin, the meat, and a broth
    made from the bones. It is considered a national dish of {country} and a symbol of
    Chinese culinary art. Peking Duck has also been a part of many state banquets and
    diplomatic events, symbolizing Chinese hospitality and culinary excellence.
    - Question: During a state banquet featuring the dish in the image, which aspect of
    its presentation is most likely emphasized to symbolize Chinese culinary excellence
    and hospitality? Options: (A) The use of exotic spices (B) The serving of the duck
    with rice (C) The incorporation of seafood (D) The carving of the duck in front of
    diners
    - Answer: (D) The carving of the duck in front of diners
    - Reason: The traditional carving of Peking Duck in front of diners highlights the
    skill involved in its preparation and serves as a symbol of Chinese culinary
    excellence and hospitality.
    Image: The White House
    Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue
    NW in Washington, D.C., is the official residence and workplace of the President of
    the United States. Construction began in 1792 and was completed in 1800. The building
    was designed by Irish-born architect James Hoban in the neoclassical style, featuring
    a white-painted Aquia Creek sandstone exterior. It has been the residence of every U.S
    . president since John Adams. The White House has undergone several renovations and
    expansions, including the addition of the West Wing, East Wing, and the Oval Office.
    The building’s iconic appearance and historical significance make it a symbol of the U
    .S. government and a site of significant historical events. The White House has been
    the location of many important decisions, meetings with foreign dignitaries, and
    addresses to the nation. It also serves as a museum of American history, housing
    numerous artifacts and pieces of art. The White House is not only a residence but also
    a working office, with various staff members ensuring the smooth operation of the
    executive branch of the U.S. government.
    - Question: During a critical diplomatic event, the President is scheduled to meet
    with several foreign dignitaries to discuss global climate initiatives. As depicted in
    the image, which room inside the building is most likely to be used for this high level diplomatic meeting? Options: (A) The Lincoln Bedroom (B) The Oval Office (C) The
    White House Kitchen (D) The East Room
    - Answer: (B) The Oval Office
    - Reason: The Oval Office is traditionally used for important meetings and discussions
    , making it the most likely choice for a high-level diplomatic meeting with foreign
    dignitaries.

    
    Here is the introduction:
    {introduction}
    
    Now, please generate the scene reasoning question for {concept} of {country}.
    """
    
    try:
        # Generate content using Google's genai
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        question = response.text  # Extract the generated text
        print(question)

        # Extract text after </think> if it exists
        match = re.search(pattern, question)
        if match:
            question = match.group(1)

        questions.append(question)
    except Exception as e:
        print(f"Error generating question for {concept}: {e}")
        questions.append("")

    time.sleep(10)  # Add a delay between requests to avoid rate limits

# Save results
questions_phase1["questions"] = pd.Series(questions)
questions_phase1.to_csv("Questions_phase_3.csv", index=False)
