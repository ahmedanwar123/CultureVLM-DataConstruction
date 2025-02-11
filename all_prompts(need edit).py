from google import genai
import pandas as pd
import re
import time

# Load data
data = pd.read_csv("Dataset_CultureVLM_style - Concepts Introductions.csv")
questions_phase2 = pd.DataFrame()
questions_phase2['concept'] = data['concept']
print(questions_phase2.head())

# Initialize Google genai Client
client = genai.Client(api_key="AIzaSyCF7xCRjmobiB15TrYAuqh4tYrOxx4vWMs")  # Replace with your actual Google API key

pattern = r"</think>([\s\S]*)"

# Define all prompts
prompts = {
    "prompt1": """
Concept: {concept}
Country: {country}
Class: {class_}
Please determine whether "{concept}" is a kind of {class_} that can reflect {country}
culture and whether it can serve as a symbol of {country} culture (unique and very
famous within {country}, and not commonly seen in other parts of the world).
Additionally, the symbol should not be a broad category that includes various specific
items, but rather a distinct and indivisible entity, such as a specific dance form, a
famous individual’s photograph, a renowned landmark, etc.
Here are some counterexamples:
- Broad concepts like "fast food" (which includes burgers, fries, etc.), "traditional
Chinese instruments" (which include guzheng, erhu, etc.) "Dance" (which include jazz
dance, Square dancing, etc.) are not acceptable due to their lack of a unified visual
marker.
- Concept itself is the wrong word, such as "...", "N/A".
- Concept exists in many regions, such as "pork", "duck" or "grapes" (which might be a
specialty or staple in certain countries but is quite common in many places. However,
specific concepts like "peking duck" or "schweinshaxe" would be correct).
Please provide your short explanation and include your answer (Yes/No) into <<<>>>.
For example, if you think "{concept}" is a specific and indivisible symbol of {country
} culture, please write <<<Yes>>>.
""",
    "prompt2": """
Please extract cultural elements from the given wikipedia document that can represent
{country} culture or are very famous in {country}, including the following categories:
### Categories
- Food: e.g., local specialty dishes, traditional festival foods.
- Plants: e.g., unique flowers, crops in {country}.
- Animals: e.g., unique wild animals, livestock in {country}.
- Famous Landmarks: e.g., famous historical sites, buildings.
- Festivals: e.g., unique festival celebration scenes.
- Historical Artifacts: e.g., museum collections, ancient relics.
- Historical Figures: e.g., portraits of historical figures, statues.
- Traditional Clothing: e.g., ethnic clothing, festival attire.
- Architectural Styles: e.g., traditional architecture, modern landmark buildings.
- Handicrafts and Artifacts: e.g., ethnic handicrafts, traditional handmade items.
- Music and Dance: e.g., traditional musical instruments, dance scenes.
- Religion and Belief: e.g., temples and churches, religious ceremonies.
- Natural Scenery and Ecosystems: e.g., unique natural landscapes, ecological reserves.
- Markets and Shopping Traditions: e.g., local markets, specialty shops.
- Entertainment and Performing Arts: e.g., theater performances, street performers.
Please note that not all categories may be included in the document. Only list the
most famous cultural elements, with a total not exceeding 10. For categories without
famous elements, use ’NA’ to indicate. Directly output in the format "Category: [
Element1, Element2, ...]", for example:
### Cultural Elements
- Food: [Food 1], [Food 2], ...
- Plants: NA
- Music and Dance: [Musical Instrument 1], [Dance Scene 2], ...
The following is the wikipedia document.
### Wikipedia Document
{wikipedia}
""",
    "prompt3": """
This is the {concept} of {country}. Your task is to generate a multiple-choice
question that asks the user to identify what is shown in the image. Use the following
format for your question:
- Question: [Your Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D)
[Option 4]
For example, if the image shows a Peking Duck of China, the question and options
should look like this:
- Question: What traditional dish is shown in this image? Options: (A) Cantonese Roast
Duck (B) Peking Duck (C) Sichuan Spicy Duck (D) Nanjing Salted Duck
If the image shows the Erhu of China:
- Question: What musical instrument is shown in this image? Options: (A) Pipa (B) Erhu
(C) Sanxian (D) Yangqin
If the image shows the White House of US:
- Question: What famous American building is shown in this image? Options: (A) The
Capitol (B) The White House (C) The Lincoln Memorial (D) The Supreme Court Building
Ensure that "{concept}" should be included in one of the options. Ensure that the
options are plausible but only one is the correct answer. The incorrect options should
be similar enough to the correct one to create a challenge, but not so similar that
they cause any potential ambiguity.
""",
    "prompt4": """
Please provide a detailed introduction of {concept} of {country}, including
information such as: Location and Features (where it is found or originates from, what
makes it unique); Time period (when it was created or became significant); History (
historical background and development, or any significant events); Cultural
significance (cultural Context in {country}, modern-day significance); Stories or
Legends (Any stories, legends, or folklore associated with {concept}).
Use the following format for your introduction:
Introduction of {concept}: [Detailed Introduction of {concept}]
Here are two examples:
Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in
Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)
and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is
known for its thin, crispy skin and is served with pancakes, hoisin sauce, and
scallions. The preparation involves inflating the duck to separate the skin from the
fat, marinating it, and roasting it in a closed or hung oven. It is considered a
national dish of China and a symbol of Chinese culinary art.
Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue
NW in Washington, D.C., is the official residence and workplace of the President of
the United States. Construction began in 1792 and was completed in 1800. The building
was designed by Irish-born architect James Hoban in the neoclassical style. It has
been the residence of every U.S. president since John Adams. The White House has
undergone several renovations and expansions, including the addition of the West Wing
and the Oval Office. It is a symbol of the U.S. government and a site of significant
historical events.
Now please provide the introduction for {concept}.
Introduction of {concept}:
""",
    "prompt5": """
This public image shows the "{concept}" of {country}. Generate a multiple-choice
question based on this image and the introduction of {concept}. Provide the correct
answer immediately following the question.
Ensure the question delves into deeper cultural knowledge but does not directly name
the {concept}. The options should be somewhat confusing to increase the difficulty,
but there must be only one correct answer. Users can only answer based on the image,
so don’t mention any "introduction" or "{concept}" in the question. Use the following
format for your generated question:
- Question: [Your Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D)
[Option 4]
- Answer: (X) [Option X]
Here are two examples:
Image: Peking Duck
Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in
Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)
and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is
known for its thin, crispy skin and is served with pancakes, hoisin sauce, and
scallions. The preparation involves inflating the duck to separate the skin from the
fat, marinating it, and roasting it in a closed or hung oven. It is considered a
national dish of China and a symbol of Chinese culinary art.
- Question: During which dynasty did the dish shown in the image become a staple in
the cuisine of its country? Options: (A) Tang Dynasty (B) Song Dynasty (C) Ming
Dynasty (D) Qing Dynasty
- Answer: (C) Ming Dynasty
Image: The White House
Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue
NW in Washington, D.C., is the official residence and workplace of the President of
the United States. Construction began in 1792 and was completed in 1800. The building
was designed by Irish-born architect James Hoban in the neoclassical style. It has
been the residence of every U.S. president since John Adams. The White House has
undergone several renovations and expansions, including the addition of the West Wing
and the Oval Office. It is a symbol of the U.S. government and a site of significant
historical events.
- Question: Who was the architect responsible for designing the building shown in the
image? Options: (A) James Hoban (B) Benjamin Latrobe (C) Thomas Jefferson (D) Charles
Bulfinch
- Answer: (A) James Hoban
Now please generate the question for the Image: {concept} of {country}
{introduction}
""",
    "prompt6": """
This public image shows the "{concept}" of {country}. Generate a visual reasoning
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
answer based on the image, so don’t mention any "introduction" or "{concept}" in the
question.
Use the following format for your introduction and question:
- Question: [Your Scenario-based Question] Options: (A) [Option 1] (B) [Option 2] (C)
[Option 3] (D) [Option 4]
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
the image, which room inside the building is most likely to be used for this highlevel diplomatic meeting? Options: (A) The Lincoln Bedroom (B) The Oval Office (C) The
White House Kitchen (D) The East Room
- Answer: (B) The Oval Office
- Reason: The Oval Office is traditionally used for important meetings and discussions
, making it the most likely choice for a high-level diplomatic meeting with foreign
dignitaries.
Now please generate the question for the Image: {concept} of {country}
{introduction}
""",
    "prompt7": """
(Hint: This image shows the {concept} of {country}.)
Here is a question about this image:
{question}
First, describe the image in detail and analyze its features. Then, analyze the
characteristics of the four options and compare each one with the features of the
image. Finally, provide your final answer. Please include your answer into (). For
example, if you choose A, please write (A).
""",
    "prompt8": """
(Hint: This image shows the {concept} of {country}. {introduction})
Here is a question about this image:
{question}
First, provide a detailed description and identification of the image, analyzing its
features. Then, conduct a comprehensive analysis of the question and four options
based on the Hint and your knowledge. Finally, present the final answer.
Please refrain from explicitly mentioning the "Hint" in your response, as these are
for your discreet knowledge and not provided by the question. Please include your
answer into (). For example, if you choose A, please write (A).
Response:
""",
    "prompt9": """
Here is a question about this image:
Question: {question} Options: {options}
First, describe and identify the image. Then, analyze the question and all four
options in detail. Finally, provide the answer, indicating your final choice in
parentheses. For example, if you choose A, please write (A).
"""
}

# Loop through each prompt and generate results
for prompt_name, prompt_template in prompts.items():
    questions = []
    for concept in data['concept']:
        introduction = data.loc[data['concept'] == concept, 'intro']
        introduction = introduction.to_list()[0]

        formatted_prompt = prompt_template.format(concept=concept, introduction=introduction)

        try:
            # Generate content using Google's genai
            response = client.models.generate_content(
                model="gemini-2.0-flash",  # Replace with your desired model
                contents=formatted_prompt
            )
            question = response.text  # Extract the generated text
            print(question)
            match = re.search(pattern, question)
            if match:
                question = match.group(1)
            questions.append(question)
        except Exception as e:
            print(f"Error generating question for {concept}: {e}")
            questions.append("")  # Append an empty string if an error occurs

        time.sleep(10)  # Add a delay between requests to avoid rate limits

    # Save results to a unique file for each prompt
    output_file = f"Questions_phase_2_{prompt_name}.csv"
    questions_phase2['questions'] = pd.Series(questions)
    questions_phase2.to_csv(output_file)
    print(f"Results saved to {output_file}")