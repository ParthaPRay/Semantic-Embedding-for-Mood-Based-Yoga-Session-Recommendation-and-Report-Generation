# Calls mood_9.py 

import requests
import json
import time

# New prompts (one per line from the provided text)
prompts = [
    "Individuals feeling anxious, nervous, or uneasy often experience persistent worry, restlessness, and a sense of impending trouble. These emotions can lead to physical symptoms like rapid heartbeat and tension, making it difficult to relax or focus on daily activities.",
    "When overwhelmed by stress, tension, or overload, a person may feel pressured, fatigued, and unable to cope with responsibilities. This can result in irritability, headaches, and difficulty concentrating, impacting both mental and physical health.",
    "Experiencing depression, sadness, or a low mood involves persistent feelings of hopelessness, lack of energy, and disinterest in activities once enjoyed.",
    "Those suffering from insomnia, sleep issues, or nighttime restlessness often face difficulty falling or staying asleep, leading to fatigue and irritability during the day.",
    "Individuals experiencing low energy, fatigue, or exhaustion feel constantly drained, lacking the motivation to perform daily tasks.",
    "A person dealing with lack of focus, distractibility, or mental fog struggles to concentrate and process information effectively.",
    "Feeling angry, frustrated, or irritable involves frequent emotional outbursts, impatience, and a short temper.",
    "Experiencing grief, sadness, or heartache encompasses deep emotional pain, longing, and sorrow. Individuals may withdraw from social interactions, struggle with daily routines, and find it hard to find joy.",
    "Those with self-esteem issues, low confidence, or insecurity often doubt their abilities, fear judgment, and avoid taking risks.",
    "Individuals feeling fear, insecurity, or apprehension experience heightened anxiety, uncertainty, and a sense of vulnerability.",
    "Feeling overwhelmed, overloaded, or burnt out involves chronic stress, emotional exhaustion, and a sense of being unable to meet demands.",
    "Those experiencing loneliness, isolation, or a lack of connection feel disconnected from others, leading to feelings of emptiness and sadness.",
    "Feeling irritable, agitated, or tense involves constant irritability, heightened sensitivity, and a restless mind.",
    "Individuals with a lack of motivation, apathy, or low drive exhibit diminished enthusiasm, interest, and initiative.",
    "Experiencing shyness, social anxiety, or timidity involves fear of judgment, reluctance to engage in social interactions, and discomfort in group settings.",
    "Those who overthink, ruminate, or experience mental overload often dwell on negative thoughts, leading to anxiety and indecision.",
    "Feeling impatient, restless, or in a hurry involves a constant desire for immediate results, difficulty waiting, and a sense of urgency.",
    "Individuals with low self-confidence, self-doubt, or insecurity frequently question their abilities, fear failure, and struggle with assertiveness.",
    "Experiencing chronic tension, muscle tightness, or stiffness results in persistent physical discomfort, reduced flexibility, and pain.",
    "Those with mood swings, emotional fluctuations, or instability experience rapid and unpredictable changes in emotions.",
    "Feeling frustrated, agitated, or irritable involves frequent feelings of annoyance, impatience, and a short temper.",
    "Individuals with emotional instability, mood disorders, or emotional turbulence experience intense and unpredictable emotional states.",
    "Experiencing restlessness, hyperactivity, or agitation involves constant movement, inability to relax, and heightened irritability.",
    "Those feeling guilt, remorse, or regret often dwell on past actions, leading to persistent self-criticism and emotional pain.",
    "Individuals experiencing shame, embarrassment, or humiliation feel intense self-consciousness, inadequacy, and a desire to hide."
]

# Generate an equal number of unrelated prompts
unrelated_prompts = [
    "How to bake a perfect chocolate cake?",
    "Tips for arranging a bouquet of flowers?",
    "What's the best way to tune a guitar?",
    "How to fix a leaking kitchen faucet?",
    "Suggestions for planning a road trip?",
    "How to improve your public speaking skills?",
    "What are some effective study techniques?",
    "How to train for a marathon?",
    "Recommendations for learning a new language?",
    "How to build a birdhouse?",
    "Ways to decorate a small apartment?",
    "How to make homemade ice cream?",
    "What are some basic programming concepts?",
    "Tips for maintaining a car engine?",
    "How to create a personal budget?",
    "Best practices for composting at home?",
    "How to improve photography skills?",
    "Tips for growing vegetables in a home garden?",
    "How to start journaling regularly?",
    "What are some simple yoga exercises?",
    "How to meditate for beginners?",
    "Ways to reduce household waste?",
    "How to choose the right running shoes?",
    "Recommendations for healthy snacking?",
    "How to get rid of clutter effectively?"
]

# Ensure unrelated_prompts has the same length as prompts
if len(unrelated_prompts) < len(prompts):
    # If fewer unrelated prompts than required, repeat until we match the count
    needed = len(prompts) - len(unrelated_prompts)
    unrelated_prompts += (unrelated_prompts * (needed // len(unrelated_prompts) + 1))[:needed]

# Truncate if we've got too many
unrelated_prompts = unrelated_prompts[:len(prompts)]

# Combine the new prompts with the unrelated prompts
test_prompts = prompts + unrelated_prompts

# API endpoint
api_url = "http://localhost:5000/process_prompt"

def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps({"prompt": prompt}))
        result = response.json()
        return (prompt, result)
    except Exception as e:
        return (prompt, f"Error: {str(e)}")

if __name__ == "__main__":
    for prompt in test_prompts:
        prompt_result, response = send_prompt(prompt)
        print("Prompt:")
        print(prompt_result)
        print("Response:")
        print(response)
        print("-" * 50)
        time.sleep(0.5)  # small delay between calls
