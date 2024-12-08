# Yoga Asana Static Router with PDF Report Generation using FastAPI and Cosine Similarity
# This application routes user mood prompts to the appropriate Yoga Asana based on semantic similarity,
# generates a personalized PDF report, and provides it for download.
# curl -X POST "http://localhost:5000/process_prompt" -H "Content-Type: application/json" -d '{"prompt": "I feel overwhelmed and stressed with my work."}'

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import requests
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import os

app = FastAPI()

# Configuration Variables
EMBED_MODEL = "all-minilm:33m"
OLLAMA_API_URL = "http://localhost:11434/api/embed"
SIMILARITY_THRESHOLD = 0.4950

# Directory to store generated PDF reports
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

class Prompt(BaseModel):
    prompt: str

# Define the Yoga Asana routes with their corresponding utterances
class YogaAsana:
    def __init__(self, name, utterances, how_to_do, frequency, timing, dietary, lifestyle, benefits):
        self.name = name
        self.utterances = utterances
        self.how_to_do = how_to_do
        self.frequency = frequency
        self.timing = timing
        self.dietary = dietary
        self.lifestyle = lifestyle
        self.benefits = benefits

# Predefined Yoga Asanas and their detailed mood issues as utterances
yoga_asanas = [
    YogaAsana(
        name="Child’s Pose (Balasana)",
        utterances=[
            "Individuals feeling anxious, nervous, or uneasy often experience persistent worry, restlessness, and a sense of impending trouble. This state hinders overall well-being and peace of mind."
        ],
        how_to_do="1. Kneel on the mat with big toes touching and knees spread apart.\n"
                  "2. Sit back on your heels and extend your arms forward, lowering your torso between your thighs.\n"
                  "3. Rest your forehead on the mat and breathe deeply.",
        frequency="3-5 days/week",
        timing="Evening",
        dietary="Light meals, Include magnesium-rich foods like spinach and almonds, Herbal teas (e.g., chamomile)",
        lifestyle="Practice deep breathing or meditation, Limit caffeine and sugar intake, Maintain a consistent sleep schedule",
        benefits="Calms the nervous system, Reduces stress, Promotes relaxation, Alleviates anxiety"
    ),
    YogaAsana(
        name="Downward Facing Dog (Adho Mukha Svanasana)",
        utterances=[
            "When overwhelmed by stress, tension, or overload, a person may feel pressured, fatigued, and unable to cope with responsibilities."
        ],
        how_to_do="1. Start on all fours with hands shoulder-width apart.\n"
                  "2. Lift your hips towards the ceiling, forming an inverted V-shape.\n"
                  "3. Keep your spine straight and heels reaching towards the mat.",
        frequency="4-6 days/week",
        timing="Morning & Evening",
        dietary="Balanced diet, Incorporate whole grains, lean proteins, and vegetables, Stay hydrated",
        lifestyle="Incorporate time management techniques, Engage in hobbies and leisure activities, Practice mindfulness and relaxation techniques",
        benefits="Relieves stress, Invigorates the body, Stretches the hamstrings, Calms the mind, Releases tension"
    ),
    YogaAsana(
        name="Camel Pose (Ustrasana)",
        utterances=[
            "Experiencing depression, sadness, or a low mood involves persistent feelings of hopelessness, lack of energy, and disinterest in activities once enjoyed."
        ],
        how_to_do="1. Kneel on the mat with knees hip-width apart.\n"
                  "2. Place your hands on your lower back for support.\n"
                  "3. Gently arch your spine backward, reaching for your heels if comfortable.\n"
                  "4. Keep your neck relaxed and breathe deeply.",
        frequency="5 days/week",
        timing="Afternoon",
        dietary="Increase intake of omega-3 fatty acids (e.g., fish, flaxseeds), Incorporate vitamin D-rich foods",
        lifestyle="Seek social support, Engage in regular physical activity, Practice gratitude journaling",
        benefits="Opens the heart, Reduces feelings of sadness, Stimulates the nervous system, Boosts mood, Enhances emotional well-being"
    ),
    YogaAsana(
        name="Corpse Pose (Savasana)",
        utterances=[
            "Those suffering from insomnia, sleep issues, or nighttime restlessness often face difficulty falling or staying asleep, leading to fatigue and irritability during the day."
        ],
        how_to_do="1. Lie flat on your back with arms at your sides, palms facing up.\n"
                  "2. Close your eyes and relax every part of your body.\n"
                  "3. Focus on your breath and remain still for 5-15 minutes.",
        frequency="Daily",
        timing="Nighttime",
        dietary="Avoid heavy meals before bedtime, Include magnesium-rich foods like bananas and nuts, Herbal teas (e.g., valerian root)",
        lifestyle="Maintain a regular sleep schedule, Create a calming bedtime routine, Limit screen time before bed",
        benefits="Promotes deep relaxation, Calms the mind, Improves circulation, Reduces anxiety, Facilitates better sleep"
    ),
    YogaAsana(
        name="Upward Facing Dog (Urdhva Mukha Svanasana)",
        utterances=[
            "Individuals experiencing low energy, fatigue, or exhaustion feel constantly drained, lacking the motivation to perform daily tasks."
        ],
        how_to_do="1. Lie face down with legs together and tops of the feet pressing into the mat.\n"
                  "2. Place your hands under your shoulders and press up, lifting your chest and thighs off the ground.\n"
                  "3. Keep your shoulders down and gaze forward.\n"
                  "4. Hold for 15-30 seconds, then release.",
        frequency="5-7 days/week",
        timing="Morning",
        dietary="Balanced, nutrient-rich meals, Include complex carbohydrates and lean proteins, Stay hydrated",
        lifestyle="Ensure adequate rest, Limit alcohol and caffeine, Engage in regular physical activity",
        benefits="Increases energy levels, Combats fatigue, Boosts confidence, Strengthens muscles, Enhances overall vitality"
    ),
    YogaAsana(
        name="Tree Pose (Vrksasana)",
        utterances=[
            "A person dealing with lack of focus, distractibility, or mental fog struggles to concentrate and process information effectively."
        ],
        how_to_do="1. Stand tall with feet together.\n"
                  "2. Shift your weight onto the left foot and place the right foot on the inner left thigh or calf (avoid the knee).\n"
                  "3. Bring your hands to prayer position at your chest or raise them overhead.\n"
                  "4. Hold for 30 seconds to 1 minute, then switch sides.",
        frequency="4-6 days/week",
        timing="Morning & Afternoon",
        dietary="Include brain-boosting foods like blueberries and nuts, Ensure adequate hydration, Limit processed foods and sugars",
        lifestyle="Practice time management, Minimize distractions, Engage in cognitive exercises like puzzles or reading",
        benefits="Enhances concentration, Improves balance, Fosters grounding, Encourages mental discipline, Enhances coordination"
    ),
    YogaAsana(
        name="Revolved Triangle Pose (Parivrtta Trikonasana)",
        utterances=[
            "Feeling angry, frustrated, or irritable involves frequent emotional outbursts, impatience, and a short temper."
        ],
        how_to_do="1. Stand with feet wide apart.\n"
                  "2. Turn the right foot out 90 degrees and the left foot slightly inward.\n"
                  "3. Extend arms to the sides and hinge at the hips to twist the torso towards the right, placing the left hand on the right shin or floor.\n"
                  "4. Extend the right arm towards the ceiling.\n"
                  "5. Hold for 30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Afternoon",
        dietary="Incorporate anti-inflammatory foods like leafy greens and turmeric, Avoid excessive caffeine and spicy foods",
        lifestyle="Practice journaling or expressive writing, Engage in calming activities like listening to music, Develop anger management strategies",
        benefits="Stimulates digestion, Enhances mental clarity, Stretches the hips and ankles, Releases tension, Fosters emotional balance"
    ),
    YogaAsana(
        name="Bridge Pose (Setu Bandhasana)",
        utterances=[
            "Experiencing grief, sadness, or heartache encompasses deep emotional pain, longing, and sorrow."
        ],
        how_to_do="1. Lie on your back with knees bent and feet flat on the floor, hip-width apart.\n"
                  "2. Press your feet and arms into the mat as you lift your hips toward the ceiling.\n"
                  "3. Clasp your hands under your back and extend the arms.\n"
                  "4. Hold for 30 seconds to 1 minute, then release slowly.",
        frequency="4-6 days/week",
        timing="Evening",
        dietary="Incorporate mood-boosting foods like dark chocolate and berries, Avoid heavy or greasy foods",
        lifestyle="Allow time for emotional expression, Seek support from friends or a therapist, Engage in creative activities",
        benefits="Reduces anxiety, Improves mood, Fosters emotional release, Calms the mind, Opens the heart"
    ),
    YogaAsana(
        name="Cobra Pose (Bhujangasana)",
        utterances=[
            "Those with self-esteem issues, low confidence, or insecurity often doubt their abilities, fear judgment, and avoid taking risks."
        ],
        how_to_do="1. Lie face down with legs together and hands under shoulders.\n"
                  "2. Press into your hands to lift your chest off the ground.\n"
                  "3. Keep elbows slightly bent and shoulders relaxed.\n"
                  "4. Hold for 15-30 seconds, then lower down.",
        frequency="4-5 days/week",
        timing="Morning & Evening",
        dietary="Include protein-rich foods to support muscle and energy levels, Incorporate foods high in B vitamins for brain health",
        lifestyle="Practice positive affirmations, Engage in confidence-building activities, Limit negative self-talk",
        benefits="Boosts self-esteem and courage, Enhances confidence, Encourages perseverance, Opens the heart, Reduces stress"
    ),
    YogaAsana(
        name="Warrior III (Virabhadrasana III)",
        utterances=[
            "Individuals feeling fear, insecurity, or apprehension experience heightened anxiety, uncertainty, and a sense of vulnerability."
        ],
        how_to_do="1. Stand on your left foot.\n"
                  "2. Hinge forward at the hips, extending the right leg straight back.\n"
                  "3. Stretch your arms forward or alongside your body.\n"
                  "4. Keep your torso parallel to the floor.\n"
                  "5. Hold for 20-30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Morning",
        dietary="Incorporate foods rich in vitamin C and E for adrenal support, Avoid excessive caffeine and alcohol",
        lifestyle="Practice self-affirmation and visualization, Engage in confidence-building activities, Limit exposure to fear-inducing stimuli",
        benefits="Enhances balance, Builds resilience, Encourages mental discipline, Fosters grace and confidence, Improves coordination"
    ),
    YogaAsana(
        name="Supine Spinal Twist (Supta Matsyendrasana)",
        utterances=[
            "Feeling overwhelmed, overloaded, or burnt out involves chronic stress, emotional exhaustion, and a sense of being unable to meet demands."
        ],
        how_to_do="1. Lie on your back with arms extended to the sides.\n"
                  "2. Bend your knees and drop them to the right while keeping shoulders on the mat.\n"
                  "3. Turn your head to the left if comfortable.\n"
                  "4. Hold for 1-2 minutes, then switch sides.",
        frequency="5 days/week",
        timing="Afternoon & Evening",
        dietary="Eat light, easily digestible meals, Include anti-inflammatory foods like turmeric and ginger",
        lifestyle="Prioritize tasks and delegate when possible, Take regular breaks during work, Practice mindfulness and stress management techniques",
        benefits="Promotes relaxation, Massages internal organs, Opens the hips, Fosters a sense of calm, Reduces stress"
    ),
    YogaAsana(
        name="Lotus Pose (Padmasana)",
        utterances=[
            "Those experiencing loneliness, isolation, or a lack of connection feel disconnected from others, leading to feelings of emptiness and sadness."
        ],
        how_to_do="1. Sit on the mat with legs extended.\n"
                  "2. Bend the right knee and place the right foot on the left thigh.\n"
                  "3. Repeat with the left knee, placing the left foot on the right thigh.\n"
                  "4. Sit tall, hands resting on your knees or in prayer position.\n"
                  "5. Hold for 5-10 minutes, focusing on your breath.",
        frequency="4-6 days/week",
        timing="Morning & Evening",
        dietary="Include foods rich in tryptophan (e.g., turkey, nuts) to boost serotonin, Stay hydrated with water and herbal teas",
        lifestyle="Engage in social activities or join groups, Practice self-compassion and self-love, Limit isolation by setting social goals",
        benefits="Enhances meditation, Cultivates inner peace, Promotes deep relaxation, Fosters emotional connection, Integrates practice benefits"
    ),
    YogaAsana(
        name="CatCow Pose (Marjaryasana-Bitilasana)",
        utterances=[
            "Feeling irritable, agitated, or tense involves constant irritability, heightened sensitivity, and a restless mind."
        ],
        how_to_do="1. Start on all fours with wrists under shoulders and knees under hips.\n"
                  "2. Inhale, arching the back (Cow Pose), lifting the head and tailbone.\n"
                  "3. Exhale, rounding the spine (Cat Pose), tucking the chin and tailbone.\n"
                  "4. Repeat for 5-10 breaths.",
        frequency="5 days/week",
        timing="Morning & Evening",
        dietary="Incorporate calming foods like chamomile tea and whole grains, Avoid spicy and processed foods",
        lifestyle="Practice patience and mindfulness, Engage in calming activities like reading or listening to music, Use stress-relief techniques",
        benefits="Promotes mindfulness, Reduces tension, Releases back and hip tension, Fosters playfulness, Calms the mind"
    ),
    YogaAsana(
        name="Locust Pose (Salabhasana)",
        utterances=[
            "Individuals with a lack of motivation, apathy, or low drive exhibit diminished enthusiasm, interest, and initiative."
        ],
        how_to_do="1. Lie face down with arms alongside the body, palms facing down.\n"
                  "2. Inhale and lift your head, chest, arms, and legs off the ground.\n"
                  "3. Keep your gaze forward and hold for 20-30 seconds.\n"
                  "4. Exhale and release.",
        frequency="5-7 days/week",
        timing="Morning",
        dietary="Include energizing foods like bananas and nuts, Eat balanced meals with complex carbohydrates",
        lifestyle="Set clear, achievable goals, Maintain a positive mindset, Limit procrastination by creating structured routines",
        benefits="Energizes the body, Strengthens the core, Enhances concentration, Boosts confidence, Increases energy levels"
    ),
    YogaAsana(
        name="Fish Pose (Matsyasana)",
        utterances=[
            "Experiencing shyness, social anxiety, or timidity involves fear of judgment, reluctance to engage in social interactions, and discomfort in group settings.",
            "Individuals experiencing shame, embarrassment, or humiliation feel intense self-consciousness, inadequacy, and a desire to hide."
        ],
        how_to_do="1. Lie on your back with legs extended and arms by your sides.\n"
                  "2. Press your forearms into the mat and lift your chest and head.\n"
                  "3. Tilt your head back, supporting it with your hands if needed.\n"
                  "4. Hold for 30 seconds to 1 minute, then release.",
        frequency="4-6 days/week",
        timing="Morning & Evening",
        dietary="Include foods that promote self-acceptance like whole grains and lean proteins, Avoid excessive sugar and processed foods",
        lifestyle="Practice self-affirmations, Engage in activities that build self-esteem, Limit negative self-talk",
        benefits="Opens the chest and lungs, Boosts confidence, Enhances balance, Fosters grace, Reduces feelings of vulnerability"
    ),
    YogaAsana(
        name="Mountain Pose (Tadasana)",
        utterances=[
            "Those who overthink, ruminate, or experience mental overload often dwell on negative thoughts, leading to anxiety and indecision."
        ],
        how_to_do="1. Stand with feet together, arms at sides.\n"
                  "2. Distribute your weight evenly across both feet.\n"
                  "3. Engage your thighs, lift your chest, and relax your shoulders.\n"
                  "4. Hold for 1-2 minutes, focusing on your breath.",
        frequency="5 days/week",
        timing="Afternoon & Evening",
        dietary="Include brain-calming foods like leafy greens and whole grains, Avoid stimulants like caffeine in the evening",
        lifestyle="Practice mindfulness and grounding techniques, Limit multitasking, Create a quiet environment for reflection",
        benefits="Enhances body awareness, Promotes grounding, Calms the mind, Opens the chest, Provides a sense of safety"
    ),
    YogaAsana(
        name="Half Moon Pose (Ardha Chandrasana)",
        utterances=[
            "Feeling impatient, restless, or in a hurry involves a constant desire for immediate results, difficulty waiting, and a sense of urgency."
        ],
        how_to_do="1. Stand with feet wide apart.\n"
                  "2. Turn the right foot out 90 degrees and the left foot slightly inward.\n"
                  "3. Shift weight onto the right foot and place the right hand on the mat or a block.\n"
                  "4. Lift the left leg parallel to the floor and extend the left arm upward.\n"
                  "5. Hold for 30 seconds to 1 minute, then switch sides.",
        frequency="3-5 days/week",
        timing="Morning",
        dietary="Incorporate foods that stabilize blood sugar like oats and legumes, Avoid sugary snacks and beverages",
        lifestyle="Practice patience and mindfulness, Break tasks into manageable steps, Engage in calming activities",
        benefits="Improves balance, Enhances coordination, Stretches the body, Strengthens legs, Encourages mental discipline"
    ),
    YogaAsana(
        name="Heart Opening Poses (Ustrasana, Bhujangasana)",
        utterances=[
            "Those feeling guilt, remorse, or regret often dwell on past actions, leading to persistent self-criticism and emotional pain."
        ],
        how_to_do="Camel Pose (Ustrasana):\n"
                  "1. Kneel on the mat with knees hip-width apart.\n"
                  "2. Place your hands on your lower back for support.\n"
                  "3. Gently arch your spine backward, reaching for your heels if comfortable.\n"
                  "4. Keep your neck relaxed and breathe deeply.\n\n"
                  "Cobra Pose (Bhujangasana):\n"
                  "1. Lie face down with legs together and hands under shoulders.\n"
                  "2. Press into your hands to lift your chest off the ground.\n"
                  "3. Keep elbows slightly bent and shoulders relaxed.\n"
                  "4. Hold for 15-30 seconds, then lower down.",
        frequency="4-6 days/week",
        timing="Evening",
        dietary="Incorporate heart-healthy foods like leafy greens and berries, Avoid excessive junk food and alcohol",
        lifestyle="Practice forgiveness and self-compassion, Engage in reflective journaling, Seek emotional support",
        benefits="Opens the heart, Promotes forgiveness, Releases feelings of guilt, Enhances self-acceptance, Encourages emotional release"
    ),
    YogaAsana(
        name="Revolved Chair Pose (Parivrtta Utkatasana)",
        utterances=[
            "Feeling frustrated, agitated, or irritable involves frequent feelings of annoyance, impatience, and a short temper."
        ],
        how_to_do="1. Stand with feet together.\n"
                  "2. Bend your knees slightly and lower into a squat.\n"
                  "3. Bring your hands to prayer position at your chest.\n"
                  "4. Twist your torso to the right, placing your left elbow outside your right knee.\n"
                  "5. Hold for 30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Afternoon",
        dietary="Incorporate detoxifying foods like green tea and leafy vegetables, Avoid processed and fried foods",
        lifestyle="Practice deep breathing and stress management, Engage in problem-solving activities, Maintain a calm environment",
        benefits="Detoxifies the body, Enhances balance, Releases tension, Fosters a sense of calm"
    ),
    YogaAsana(
        name="Sphinx Pose (Salamba Bhujangasana)",
        utterances=[
            "Experiencing chronic tension, muscle tightness, or stiffness results in persistent physical discomfort, reduced flexibility, and pain."
        ],
        how_to_do="1. Lie on your stomach with legs extended and tops of the feet pressing into the mat.\n"
                  "2. Place your forearms on the mat, elbows under shoulders.\n"
                  "3. Lift your chest off the ground, keeping your neck neutral.\n"
                  "4. Hold for 1-2 minutes, breathing deeply.",
        frequency="5-7 days/week",
        timing="Throughout the Day",
        dietary="Incorporate stretching and hydration, Include anti-inflammatory foods like leafy greens and berries",
        lifestyle="Practice regular stretching breaks, Maintain ergonomic workspace, Use relaxation techniques like deep breathing",
        benefits="Strengthens the spine, Calms the mind, Relieves tension, Promotes flexibility, Releases neck and shoulder tension"
    ),
]

# Cache to store embeddings of utterances
cached_embeddings = {}

def get_embedding(text, model=EMBED_MODEL):
    """
    Fetches the embedding vector for the given text from the Ollama API.
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "input": text}
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json["embeddings"][0]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.
    """
    if vec1 is None or vec2 is None:
        return -1
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_best_asana(prompt_embedding, asanas):
    """
    Finds the best matching Yoga Asana based on the highest cosine similarity.
    """
    best_asana = None
    best_similarity = -1

    for asana in asanas:
        for utterance in asana.utterances:
            utterance_embedding = cached_embeddings.get(utterance)
            if utterance_embedding is None:
                # If not cached, fetch embedding
                utterance_embedding = get_embedding(utterance)
                if utterance_embedding is not None:
                    cached_embeddings[utterance] = utterance_embedding
                    print(f"Cached embedding for {asana.name}: '{utterance}'")
                else:
                    print(f"Failed to cache embedding for {asana.name}: '{utterance}'")
                    continue
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            print(f"Comparing with {asana.name}: Similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_asana = asana

    return best_asana, best_similarity

def generate_pdf_report(asana: YogaAsana, user_prompt: str, similarity: float):
    """
    Generates a PDF report for the recommended Yoga Asana.
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{asana.name.replace(' ', '_').replace('/', '_')}_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    try:
        c = canvas.Canvas(filepath, pagesize=LETTER)
        width, height = LETTER
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 50, "Personalized Yoga Asana Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 120, f"User Prompt: {user_prompt}")
        c.drawString(50, height - 140, f"Similarity Score: {similarity:.4f}")

        # Asana Name
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 180, f"Recommended Asana: {asana.name}")

        # How to Do
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 210, "How to Do:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 230)
        for line in asana.how_to_do.split('\n'):
            text.textLine(line)
        c.drawText(text)

        # Frequency
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 300, "Frequency:")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 320, asana.frequency)

        # Timing
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 340, "Timing:")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 360, asana.timing)

        # Dietary Recommendations
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 380, "Dietary Recommendations:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 400)
        for line in asana.dietary.split(','):
            text.textLine(f"- {line.strip()}")
        c.drawText(text)

        # Lifestyle Recommendations
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 480, "Lifestyle Recommendations:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 500)
        for line in asana.lifestyle.split(','):
            text.textLine(f"- {line.strip()}")
        c.drawText(text)

        # Benefits
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 580, "Benefits:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 600)
        for line in asana.benefits.split(','):
            text.textLine(f"- {line.strip()}")
        c.drawText(text)

        c.showPage()
        c.save()
        print(f"Generated PDF report: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None

@app.on_event("startup")
def startup_event():
    """
    Event that runs on application startup to verify embedding model loading and cache utterance embeddings.
    """
    test_text = "This is a test to verify embedding model."
    print(f"Attempting to load embedding model '{EMBED_MODEL}'...")
    test_embedding = get_embedding(test_text)
    if test_embedding is not None:
        print(f"Embedding model '{EMBED_MODEL}' loaded successfully.")
    else:
        print(f"Failed to load embedding model '{EMBED_MODEL}'.")

    print("Caching embeddings for Yoga Asana utterances...")
    for asana in yoga_asanas:
        for utterance in asana.utterances:
            embedding = get_embedding(utterance)
            if embedding is not None:
                cached_embeddings[utterance] = embedding
                print(f"Cached embedding for {asana.name}: '{utterance}'")
            else:
                print(f"Failed to cache embedding for {asana.name}: '{utterance}'")

@app.post("/process_prompt")
async def process_prompt(prompt: Prompt):
    """
    Processes the user's mood prompt and returns the recommended Yoga Asana along with a link to download the PDF report.
    """
    start_time = time.time()
    user_prompt = prompt.prompt
    print(f"Received prompt: {user_prompt}")

    try:
        # Get embedding for the user's prompt
        prompt_embedding = get_embedding(user_prompt)
        if prompt_embedding is None:
            print("Failed to retrieve embedding for the prompt.")
            raise HTTPException(status_code=500, detail="Failed to retrieve embedding for the prompt.")

        # Find the best matching Yoga Asana
        best_asana, similarity = find_best_asana(prompt_embedding, yoga_asanas)

        if best_asana is None or similarity < SIMILARITY_THRESHOLD:
            print("No suitable Yoga Asana found for the given prompt.")
            return {
                "status": "no_match",
                "message": "No suitable Yoga Asana found for your current mood."
            }
        else:
            print(f"Selected Yoga Asana: {best_asana.name} with similarity: {similarity:.4f}")
            # Generate PDF report
            pdf_path = generate_pdf_report(best_asana, user_prompt, similarity)
            if pdf_path is None:
                raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

            # Create a download URL (assuming the server is accessible at localhost:5000)
            download_url = f"/download_report/{os.path.basename(pdf_path)}"

            return {
                "status": "success",
                "recommended_asana": best_asana.name,
                "similarity_score": round(similarity, 4),
                "how_to_do": best_asana.how_to_do,
                "frequency_of_yoga_asana": best_asana.frequency,
                "timing_of_yoga_asana": best_asana.timing,
                "dietary_recommendations": best_asana.dietary,
                "lifestyle_recommendations": best_asana.lifestyle,
                "benefits_of_yoga_asana": best_asana.benefits,
                "download_report_url": download_url,
                "total_response_time_sec": round(time.time() - start_time, 4)
            }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the prompt.")

@app.get("/download_report/{report_filename}")
async def download_report(report_filename: str):
    """
    Endpoint to download the generated PDF report.
    """
    filepath = os.path.join(REPORTS_DIR, report_filename)
    if os.path.exists(filepath):
        print(f"Serving PDF report: {filepath}")
        return FileResponse(path=filepath, filename=report_filename, media_type='application/pdf')
    else:
        print(f"Report file not found: {filepath}")
        raise HTTPException(status_code=404, detail="Report file not found.")

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
