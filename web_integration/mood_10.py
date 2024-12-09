# Yoga Asana Static Router with PDF Report Generation using FastAPI and Cosine Similarity
# Enhanced with Ollama LLM Chat API integration for generating final comments.
# This application routes user mood prompts to the appropriate Yoga Asana based on semantic similarity,
# generates a personalized PDF report with LLM-generated comments, and provides it for download.
# CSV for metrics
# no_route when no similarity match
# With databse
# Usage:
# curl -X POST "http://localhost:5000/process_prompt" -H "Content-Type: application/json" -d '{"prompt": "I feel overwhelmed and stressed with my work."}' 
# Embed model:                 |        Threshold: 
# all-minilm:33m               |          0.4950
# nomic-embed-text             |          0.63
# snowflake-arctic-embed:110m  |          0.66
# mxbai-embed-large            |          0.62

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
import os
import json
import logging
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Configuration Variables
EMBED_MODEL = "all-minilm:33m"   ### Change embed model
OLLAMA_EMBED_API_URL = "http://localhost:11434/api/embed"
OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"  # Ollama Chat API Endpoint
LLM_MODEL_NAME = "qwen2.5:0.5b-instruct"
SIMILARITY_THRESHOLD = 0.62  ### Change it as per embed model

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

CSV_FILE = "yoga_asana_metrics.csv"
CSV_COLUMNS = [
    "datetime", "model_name", "embed_model", "prompt", "cosine_similarity_score",
    "best_asana_selected", "total_duration", "load_duration", "prompt_eval_count",
    "prompt_eval_duration", "eval_count", "eval_duration", "tokens_per_second",
    "pdf_report_time", "network_latency", "embed_match_duration", "total_response_time_sec"
]

# SQLite Database Setup
DATABASE_URL = "sqlite:///./yoga_interactions.db"

Base = declarative_base()

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prompt = Column(String, nullable=False)
    final_response = Column(String, nullable=True)
    pdf_file_path = Column(String, nullable=True)

# Create engine and session
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # Needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

class Prompt(BaseModel):
    prompt: str
    
# Pydantic models
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


    
    
# Predefined Yoga Asanas
# Revised Yoga Asanas with Enhanced Utterances
# Revised Yoga Asanas with Enhanced Utterances
yoga_asanas = [
    YogaAsana(
        name="Child Pose (Balasana)",
        utterances=[
            "I feel anxious and uneasy.",
            "Nervousness is overwhelming me.",
            "Persistent worry makes me restless.",
            "Feeling a sense of impending trouble.",
            "Experiencing anxiety and nervous tension."
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
            "I'm stressed and feeling overloaded.",
            "Tension in my body makes me anxious.",
            "Feeling overwhelmed by my responsibilities.",
            "Constant stress is weighing me down.",
            "Dealing with high levels of stress and tension."
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
            "I'm feeling depressed and sad.",
            "Low mood is affecting my daily life.",
            "Feeling hopeless and lacking energy.",
            "Experiencing persistent sadness and remorse.",
            "Dealing with depression and regret."
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
            "I'm struggling with insomnia.",
            "Having trouble sleeping at night.",
            "Feeling restless when trying to sleep.",
            "Unable to fall asleep easily.",
            "Sleep issues are affecting my health."
        ],
        how_to_do="1. Lie flat on your back with legs extended and arms at your sides, palms facing up.\n"
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
            "I have low energy and feel exhausted.",
            "Feeling fatigued all the time.",
            "Lacking confidence and self-doubt me.",
            "Insecure and drained of energy.",
            "Exhausted and lacking self-confidence."
        ],
        how_to_do="1. Lie face down with legs together and hands under shoulders.\n"
                  "2. Press into your hands to lift your chest and thighs off the ground.\n"
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
            "I'm struggling with focus and concentration.",
            "Distractibility is affecting my productivity.",
            "Feeling mentally foggy and unfocused.",
            "Difficulty concentrating on tasks.",
            "Lack of focus is hindering my work."
        ],
        how_to_do="1. Stand tall with feet together.\n"
                  "2. Shift your weight onto the left foot and place the right foot on the inner left thigh or calf (avoid the knee).\n"
                  "3. Bring your hands to prayer position at your chest or raise them overhead.\n"
                  "4. Hold for 30 seconds to 1 minute, then switch sides.",
        frequency="4-6 days/week",
        timing="Morning & Afternoon",
        dietary="Include brain-boosting foods like blueberries and nuts, Ensure adequate hydration, Limit processed foods and sugars",
        lifestyle="Practice time management and focus techniques, Minimize distractions in your environment, Engage in cognitive exercises like puzzles or reading",
        benefits="Enhances concentration, Improves balance, Fosters grounding, Encourages mental discipline, Enhances coordination"
    ),
    YogaAsana(
        name="Revolved Triangle Pose (Parivrtta Trikonasana)",
        utterances=[
            "I'm feeling angry and frustrated.",
            "Irritability is affecting my mood.",
            "Dealing with constant frustration.",
            "Feeling short-tempered and angry.",
            "Anger makes me restless."
        ],
        how_to_do="1. Stand with feet wide apart.\n"
                  "2. Turn the right foot out 90 degrees and the left foot slightly inward.\n"
                  "3. Extend arms to the sides and hinge at the hips to twist the torso towards the right, placing the left elbow outside the right knee.\n"
                  "4. Hold for 30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Afternoon",
        dietary="Incorporate detoxifying foods like green tea and leafy vegetables, Avoid processed and fried foods",
        lifestyle="Practice deep breathing and stress management techniques, Engage in problem-solving activities, Maintain a calm and organized environment",
        benefits="Detoxifies the body, Enhances balance, Releases tension, Fosters a sense of calm, Encourages emotional balance"
    ),
    YogaAsana(
        name="Bridge Pose (Setu Bandhasana)",
        utterances=[
            "I'm experiencing deep grief and sadness.",
            "Feeling heartache and emotional instability.",
            "Mood swings are making me uneasy.",
            "Dealing with emotional fluctuations and sadness.",
            "Sadness and mood instability are overwhelming."
        ],
        how_to_do="1. Lie on your back with knees bent and feet flat on the floor, hip-width apart.\n"
                  "2. Press your feet and arms into the mat as you lift your hips toward the ceiling.\n"
                  "3. Clasp your hands under your back and extend the arms.\n"
                  "4. Hold for 30 seconds to 1 minute, then release slowly.",
        frequency="4-6 days/week",
        timing="Evening",
        dietary="Include mood-boosting foods like dark chocolate and berries, Avoid excessive junk food and alcohol",
        lifestyle="Allow time for emotional expression, Seek support from friends or a therapist, Engage in creative activities",
        benefits="Strengthens the body, Reduces anxiety, Enhances mood, Fosters emotional release, Promotes relaxation"
    ),
    YogaAsana(
        name="Cobra Pose (Bhujangasana)",
        utterances=[
            "I have low self-esteem and confidence.",
            "Feeling insecure and doubtful about myself.",
            "Struggling with self-confidence issues.",
            "Insecurity is holding me back.",
            "Lacking confidence in my abilities."
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
            "I'm feeling fearful and insecure.",
            "Apprehension is affecting my confidence.",
            "Dealing with fear and insecurity.",
            "Insecurity makes me anxious.",
            "Fear is holding me back."
        ],
        how_to_do="1. Stand on your left foot.\n"
                  "2. Hinge forward at the hips, extending the right leg straight back.\n"
                  "3. Stretch your arms forward or alongside your body.\n"
                  "4. Keep your torso parallel to the floor.\n"
                  "5. Hold for 20-30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Morning",
        dietary="Incorporate foods that support nervous system health like leafy greens and nuts, Avoid excessive caffeine and sugar",
        lifestyle="Engage in resilience-building activities, Practice mindfulness and grounding techniques, Limit exposure to fear-inducing stimuli",
        benefits="Enhances balance, Builds resilience, Encourages mental discipline, Fosters grace and confidence, Improves coordination"
    ),
    YogaAsana(
        name="Supine Spinal Twist (Supta Matsyendrasana)",
        utterances=[
            "I'm feeling overwhelmed and burnt out.",
            "Overload is making me exhausted.",
            "Burnout is affecting my well-being.",
            "Feeling overloaded with responsibilities.",
            "Overwhelmed and burnt out from work."
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
            "I'm feeling lonely and isolated.",
            "Lack of connection is making me sad.",
            "Emotional instability is affecting me.",
            "Dealing with mood disorders and turbulence.",
            "Loneliness and emotional turmoil are overwhelming."
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
        name="CatCow Pose (MarjaryasanaBitilasana)",
        utterances=[
            "I'm feeling irritable and tense.",
            "Agitation is making me restless.",
            "Dealing with constant tension and irritation.",
            "Feeling tense and easily agitated.",
            "Irritability and tension are affecting me."
        ],
        how_to_do="1. Start on all fours with wrists under shoulders and knees under hips.\n"
                  "2. Inhale, arching the back (Cow Pose), lifting the head and tailbone.\n"
                  "3. Exhale, rounding the spine (Cat Pose), tucking the chin and tailbone.\n"
                  "4. Repeat for 5-10 breaths.",
        frequency="5 days/week",
        timing="Morning & Evening",
        dietary="Incorporate stretching and hydration, Include anti-inflammatory foods like leafy greens and berries",
        lifestyle="Practice regular stretching breaks, Maintain ergonomic workspace, Use relaxation techniques like deep breathing",
        benefits="Promotes mindfulness, Reduces tension, Releases back and hip tension, Fosters playfulness, Calms the mind"
    ),
    YogaAsana(
        name="Locust Pose (Salabhasana)",
        utterances=[
            "I lack motivation and feel apathetic.",
            "Low drive is affecting my productivity.",
            "Feeling apathetic and unmotivated.",
            "Struggling with low drive and motivation.",
            "Apathy makes it hard to stay active."
        ],
        how_to_do="1. Lie face down with legs together and arms alongside the body, palms facing down.\n"
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
            "I'm feeling shy and socially anxious.",
            "Dealing with timidity and embarrassment.",
            "Feeling ashamed and humiliated.",
            "Social anxiety makes me nervous.",
            "Shyness and embarrassment are overwhelming."
        ],
        how_to_do="1. Lie on your back with legs extended and arms by your sides.\n"
                  "2. Press your forearms into the mat and lift your chest and head.\n"
                  "3. Tilt your head back, supporting it with your hands if needed.\n"
                  "4. Hold for 30 seconds to 1 minute, then release.",
        frequency="4-6 days/week",
        timing="Morning & Evening",
        dietary="Include foods that promote self-acceptance like whole grains and lean proteins, Avoid excessive sugar and processed foods",
        lifestyle="Practice self-affirmations, Engage in confidence-building activities, Limit negative self-talk",
        benefits="Opens the chest and lungs, Boosts confidence, Enhances balance, Fosters grace, Reduces feelings of vulnerability"
    ),
    YogaAsana(
        name="Mountain Pose (Tadasana)",
        utterances=[
            "I'm overthinking and mentally overloaded.",
            "Rumination is affecting my peace of mind.",
            "Mental overload makes me anxious.",
            "Struggling with overthinking and ruminating.",
            "Mental fog from overthinking is overwhelming."
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
            "I'm feeling impatient and restless.",
            "Haste makes me anxious.",
            "Dealing with restlessness and impatience.",
            "Impatience is affecting my focus.",
            "Feeling hurried and restless."
        ],
        how_to_do="1. Stand with feet wide apart.\n"
                  "2. Turn the right foot out 90 degrees and the left foot slightly inward.\n"
                  "3. Shift weight onto the right foot and place your right hand on the mat or a block.\n"
                  "4. Lift the left leg parallel to the floor and extend the left arm upward.\n"
                  "5. Hold for 30 seconds to 1 minute, then switch sides.",
        frequency="3-5 days/week",
        timing="Morning",
        dietary="Incorporate foods that stabilize blood sugar like oats and legumes, Avoid sugary snacks and beverages",
        lifestyle="Practice patience and mindfulness, Break tasks into manageable steps, Engage in calming activities",
        benefits="Improves balance, Enhances coordination, Stretches the body, Strengthens legs, Encourages mental discipline"
    ),
    YogaAsana(
        name="HeartOpening Poses (Ustrasana, Bhujangasana)",
        utterances=[
            "I'm feeling guilty and remorseful.",
            "Regret is weighing me down emotionally.",
            "Dealing with shame and embarrassment.",
            "Feeling humiliated and self-conscious.",
            "Guilt and regret are affecting my well-being."
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
            "I'm feeling frustrated and agitated.",
            "Irritability is affecting my mood.",
            "Dealing with constant frustration and agitation.",
            "Feeling tense and irritable.",
            "Frustration makes me restless."
        ],
        how_to_do="1. Stand with feet together.\n"
                  "2. Bend your knees slightly and lower into a squat.\n"
                  "3. Bring your hands to prayer position at your chest.\n"
                  "4. Twist your torso to the right, placing your left elbow outside your right knee.\n"
                  "5. Hold for 30 seconds, then switch sides.",
        frequency="3-5 days/week",
        timing="Afternoon",
        dietary="Incorporate detoxifying foods like green tea and leafy vegetables, Avoid processed and fried foods",
        lifestyle="Practice deep breathing and stress management techniques, Engage in problem-solving activities, Maintain a calm and organized environment",
        benefits="Detoxifies the body, Enhances balance, Releases tension, Fosters a sense of calm"
    ),
    YogaAsana(
        name="WindRelieving Pose (Pavanamuktasana)",
        utterances=[
            "I'm feeling restless and hyperactive.",
            "Agitation makes it hard to relax.",
            "Dealing with restlessness and hyperactivity.",
            "Feeling agitated and unable to calm down.",
            "Restlessness is affecting my focus."
        ],
        how_to_do="1. Lie on your back with legs extended.\n"
                  "2. Hug your knees into your chest.\n"
                  "3. Gently rock side to side to massage the spine.\n"
                  "4. Hold for 1-2 minutes, breathing deeply.",
        frequency="5 days/week",
        timing="Throughout the Day",
        dietary="Incorporate foods that aid digestion like ginger and fennel, Stay hydrated with water and herbal teas",
        lifestyle="Practice regular relaxation breaks, Engage in calming activities like reading or listening to music, Use stress-relief techniques",
        benefits="Promotes relaxation, Aids digestion, Releases spinal tension, Calms the mind, Reduces physical agitation"
    ),
    YogaAsana(
        name="Sphinx Pose (Salamba Bhujangasana)",
        utterances=[
            "I'm experiencing chronic muscle tension.",
            "Feeling stiff and tense all the time.",
            "Dealing with persistent muscle tightness.",
            "Chronic tension is affecting my posture.",
            "Muscle stiffness makes me uncomfortable."
        ],
        how_to_do="1. Lie on your stomach with legs extended and forearms on the mat, elbows under shoulders.\n"
                  "2. Press into your forearms to lift your chest off the ground.\n"
                  "3. Keep your shoulders relaxed and gaze forward.\n"
                  "4. Hold for 1-2 minutes, breathing deeply.",
        frequency="5-7 days/week",
        timing="Throughout the Day",
        dietary="Incorporate stretching and hydration, Include anti-inflammatory foods like leafy greens and berries",
        lifestyle="Practice regular stretching breaks, Maintain ergonomic workspace, Use relaxation techniques like deep breathing",
        benefits="Strengthens the spine, Calms the mind, Relieves tension, Promotes flexibility, Releases neck and shoulder tension"
    ),
     # Add other Yoga Asanas here...
]

   


# Cache to store embeddings of utterances
cached_embeddings = {}

def get_embedding(text, model=EMBED_MODEL):
    try:
        logger.debug(f"Fetching embedding for text: {text}")
        response = requests.post(
            OLLAMA_EMBED_API_URL,
            json={"model": model, "input": text}
        )
        response.raise_for_status()
        response_json = response.json()
        embedding = response_json.get("embeddings", [])[0]
        return embedding
    except Exception as e:
        logger.error(f"Error fetching embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return -1
    try:
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return -1

def find_best_asana(prompt_embedding, asanas):
    best_asana = None
    best_similarity = -1
    for asana in asanas:
        for utterance in asana.utterances:
            utterance_embedding = cached_embeddings.get(utterance)
            if utterance_embedding is None:
                utterance_embedding = get_embedding(utterance)
                if utterance_embedding is not None:
                    cached_embeddings[utterance] = utterance_embedding
                else:
                    continue
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_asana = asana
    return best_asana, best_similarity

def generate_final_comment(asana: YogaAsana, user_prompt: str):
    few_shot_examples = [
        {
            "role": "system",
            "content": "You are an assistant that provides concise, insightful comments based on Yoga Asana recommendations."
        },
        {
            "role": "user",
            "content": "Recommended Asana: Downward Facing Dog (Adho Mukha Svanasana)\nHow to Do: ...\nGenerate a concise final comment."
        },
        {
            "role": "assistant",
            "content": "Incorporating Downward Facing Dog into your routine can significantly alleviate stress..."
        },
    ]

    user_message = (
        f"Recommended Asana: {asana.name}\n"
        f"How to Do: {asana.how_to_do}\n"
        f"Benefits: {asana.benefits}\n"
        f"Generate a concise 3-4 line final comment."
    )

    messages = few_shot_examples + [
        {
            "role": "user",
            "content": user_message
        }
    ]

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    try:
        request_start = time.time()
        response = requests.post(
            OLLAMA_CHAT_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        request_end = time.time()
        network_latency = request_end - request_start

        response.raise_for_status()
        response_json = response.json()

        total_duration = response_json.get("total_duration", 0)
        load_duration = response_json.get("load_duration", 0)
        prompt_eval_count = response_json.get("prompt_eval_count", 0)
        prompt_eval_duration = response_json.get("prompt_eval_duration", 0)
        eval_count = response_json.get("eval_count", 0)
        eval_duration = response_json.get("eval_duration", 0)

        final_comment = response_json.get("message", {}).get("content", "").strip()

        metrics = {
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "network_latency": network_latency
        }

        return final_comment, metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama LLM Chat API: {e}")
        return "Unable to generate final comment at this time.", {
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
            "network_latency": 0
        }
    except (KeyError, TypeError) as e:
        logger.error(f"Unexpected response structure from Ollama Chat API: {e}")
        return "Unable to generate final comment at this time.", {
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
            "network_latency": 0
        }

def generate_pdf_report(asana: YogaAsana, user_prompt: str, similarity: float, final_comment: str):
    pdf_start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    sanitized_asana_name = asana.name.replace(' ', '_').replace('/', '_').replace('’', '').replace("'", "")
    filename = f"{sanitized_asana_name}_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    try:
        # Create a document template
        doc = SimpleDocTemplate(
            filepath,
            pagesize=LETTER,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=20,
            spaceAfter=20,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6
        )

        normal_style = styles['BodyText']
        normal_style.fontName = 'Helvetica'
        normal_style.fontSize = 12
        normal_style.leading = 14  # Increase leading for better line spacing

        # Build the story with flowables
        story = []

        # Title
        story.append(Paragraph("Mood-Based Yoga Session Recommendation Report", title_style))
        story.append(Spacer(1, 0.2*inch))

        # Date and Prompt Info
        date_str = f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        user_prompt_str = f"User Prompt: {user_prompt}"
        similarity_str = f"Similarity Score: {similarity:.4f}"

        story.append(Paragraph(date_str, normal_style))
        story.append(Paragraph(user_prompt_str, normal_style))
        story.append(Paragraph(similarity_str, normal_style))
        story.append(Spacer(1, 0.3*inch))

        # Recommended Asana
        story.append(Paragraph(f"Recommended Asana: {asana.name}", heading_style))

        # How to Do
        story.append(Paragraph("How to Do:", heading_style))
        # Convert how_to_do text to paragraphs to handle line wrapping
        for line in asana.how_to_do.split('\n'):
            story.append(Paragraph(line.strip(), normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Frequency
        story.append(Paragraph("Frequency:", heading_style))
        story.append(Paragraph(asana.frequency, normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Timing
        story.append(Paragraph("Timing:", heading_style))
        story.append(Paragraph(asana.timing, normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Dietary Recommendations
        story.append(Paragraph("Dietary Recommendations:", heading_style))
        for item in asana.dietary.split(','):
            story.append(Paragraph(f"- {item.strip()}", normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Lifestyle Recommendations
        story.append(Paragraph("Lifestyle Recommendations:", heading_style))
        for item in asana.lifestyle.split(','):
            story.append(Paragraph(f"- {item.strip()}", normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Benefits
        story.append(Paragraph("Benefits:", heading_style))
        for item in asana.benefits.split(','):
            story.append(Paragraph(f"- {item.strip()}", normal_style))
        story.append(Spacer(1, 0.3*inch))

        # Final Comment
        story.append(Paragraph("Final Comment:", heading_style))
        # The final_comment might be multiple lines, so just one paragraph:
        story.append(Paragraph(final_comment, normal_style))

        # Build the PDF
        doc.build(story)

        pdf_end_time = time.time()
        pdf_report_time = pdf_end_time - pdf_start_time
        return filepath, pdf_report_time
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None, 0

@app.on_event("startup")
def startup_event():
    logger.info(f"Loading embedding model '{EMBED_MODEL}'...")
    test_text = "This is a test to verify embedding model."
    test_embedding = get_embedding(test_text)
    if test_embedding is not None:
        logger.info(f"Embedding model '{EMBED_MODEL}' loaded successfully.")
    else:
        logger.error(f"Failed to load embedding model '{EMBED_MODEL}'.")

    logger.info("Caching embeddings for Yoga Asana utterances...")
    for asana in yoga_asanas:
        for utterance in asana.utterances:
            if utterance not in cached_embeddings:
                embedding = get_embedding(utterance)
                if embedding is not None:
                    cached_embeddings[utterance] = embedding
                else:
                    logger.warning(f"Failed to cache embedding for {asana.name}: '{utterance}'")

@app.post("/process_prompt")
async def process_prompt(prompt: Prompt):
    start_time = time.time()
    user_prompt = prompt.prompt
    logger.info(f"Received prompt: {user_prompt}")

    # Initialize database session
    db = SessionLocal()

    try:
        prompt_embedding = get_embedding(user_prompt)
        if prompt_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve embedding for the prompt.")

        # Measure embedding match duration
        embed_start_ns = time.time_ns()
        best_asana, similarity = find_best_asana(prompt_embedding, yoga_asanas)
        embed_end_ns = time.time_ns()
        embed_match_duration = embed_end_ns - embed_start_ns

        if best_asana is None or similarity < SIMILARITY_THRESHOLD:
            # No match scenario: log with "no_route"
            log_data = {
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": LLM_MODEL_NAME,
                "embed_model": EMBED_MODEL,
                "prompt": user_prompt,
                "cosine_similarity_score": similarity if similarity != -1 else 0,
                "best_asana_selected": "no_route",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
                "tokens_per_second": 0,
                "pdf_report_time": 0,
                "network_latency": 0,
                "embed_match_duration": embed_match_duration,
                "total_response_time_sec": 0  # No match scenario
            }

            # Log to CSV
            file_exists = os.path.isfile(CSV_FILE)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

            # Log to SQLite
            interaction = UserInteraction(
                prompt=user_prompt,
                final_response="No suitable Yoga Asana found for your current mood.",
                pdf_file_path=None
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            return {
                "status": "no_match",
                "message": "No suitable Yoga Asana found for your current mood."
            }
        else:
            final_comment, llm_metrics = generate_final_comment(best_asana, user_prompt)
            if not llm_metrics:
                llm_metrics = {
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "network_latency": 0
                }

            pdf_path, pdf_report_time = generate_pdf_report(best_asana, user_prompt, similarity, final_comment)
            if pdf_path is None:
                raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

            download_url = f"/download_report/{os.path.basename(pdf_path)}"
            response_time = round(time.time() - start_time, 4)

            eval_count = llm_metrics.get("eval_count", 0)
            eval_duration = llm_metrics.get("eval_duration", 1)  # avoid division by zero
            tokens_per_second = 0
            if eval_count and eval_duration:
                tokens_per_second = (eval_count / eval_duration) * 1e9

            log_data = {
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": LLM_MODEL_NAME,
                "embed_model": EMBED_MODEL,
                "prompt": user_prompt,
                "cosine_similarity_score": similarity,
                "best_asana_selected": best_asana.name,
                "total_duration": llm_metrics.get("total_duration", 0),
                "load_duration": llm_metrics.get("load_duration", 0),
                "prompt_eval_count": llm_metrics.get("prompt_eval_count", 0),
                "prompt_eval_duration": llm_metrics.get("prompt_eval_duration", 0),
                "eval_count": llm_metrics.get("eval_count", 0),
                "eval_duration": llm_metrics.get("eval_duration", 0),
                "tokens_per_second": tokens_per_second,
                "pdf_report_time": pdf_report_time if pdf_report_time else 0,
                "network_latency": llm_metrics.get("network_latency", 0),
                "embed_match_duration": embed_match_duration,
                "total_response_time_sec": response_time  # Success scenario
            }

            # Log to CSV
            file_exists = os.path.isfile(CSV_FILE)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

            # Log to SQLite
            interaction = UserInteraction(
                prompt=user_prompt,
                final_response=final_comment,
                pdf_file_path=download_url
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)

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
                "final_comment": final_comment,
                "download_report_url": download_url,
                "total_response_time_sec": response_time
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the prompt.")

@app.get("/download_report/{report_filename}")
async def download_report(report_filename: str):
    filepath = os.path.join(REPORTS_DIR, report_filename)
    if os.path.exists(filepath):
        return FileResponse(path=filepath, filename=report_filename, media_type='application/pdf')
    else:
        raise HTTPException(status_code=404, detail="Report file not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)        