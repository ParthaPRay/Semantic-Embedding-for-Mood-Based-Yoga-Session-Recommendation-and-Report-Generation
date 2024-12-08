# Calls mood_7.py 

import requests
import json
import time

# New prompts (one per line from the provided text)
prompts = [
    "Individuals feeling anxious, nervous, or uneasy often experience persistent worry, restlessness, and a sense of impending trouble. These emotions can lead to physical symptoms like rapid heartbeat and tension, making it difficult to relax or focus on daily activities. This state hinders overall well-being and peace of mind.",
    "When overwhelmed by stress, tension, or overload, a person may feel pressured, fatigued, and unable to cope with responsibilities. This can result in irritability, headaches, and difficulty concentrating, impacting both mental and physical health. Chronic stress undermines productivity and personal relationships.",
    "Experiencing depression, sadness, or a low mood involves persistent feelings of hopelessness, lack of energy, and disinterest in activities once enjoyed. Individuals may struggle with self-esteem, experience changes in appetite or sleep, and find it challenging to engage socially, significantly affecting their quality of life.",
    "Those suffering from insomnia, sleep issues, or nighttime restlessness often face difficulty falling or staying asleep, leading to fatigue and irritability during the day. This lack of restful sleep can impair cognitive function, reduce productivity, and negatively impact emotional stability and overall health.",
    "Individuals experiencing low energy, fatigue, or exhaustion feel constantly drained, lacking the motivation to perform daily tasks. This persistent tiredness can lead to decreased productivity, impaired concentration, and a diminished ability to enjoy activities, affecting both personal and professional aspects of life.",
    "A person dealing with lack of focus, distractibility, or mental fog struggles to concentrate and process information effectively. This can result in decreased productivity, mistakes in tasks, and frustration. Cognitive functions feel sluggish, making it difficult to maintain clarity and achieve goals.",
    "Feeling angry, frustrated, or irritable involves frequent emotional outbursts, impatience, and a short temper. These emotions can strain relationships, reduce patience in stressful situations, and lead to conflicts. Persistent irritability affects mental well-being and overall mood stability.",
    "Experiencing grief, sadness, or heartache encompasses deep emotional pain, longing, and sorrow. Individuals may withdraw from social interactions, struggle with daily routines, and find it hard to find joy. This emotional turmoil profoundly impacts mental health and personal resilience.",
    "Those with self-esteem issues, low confidence, or insecurity often doubt their abilities, fear judgment, and avoid taking risks. This lack of self-assurance can hinder personal growth, limit social interactions, and affect career opportunities, leading to a diminished sense of self-worth.",
    "Individuals feeling fear, insecurity, or apprehension experience heightened anxiety, uncertainty, and a sense of vulnerability. These emotions can prevent them from taking necessary actions, reduce their willingness to engage in new experiences, and negatively impact decision-making processes.",
    "Feeling overwhelmed, overloaded, or burnt out involves chronic stress, emotional exhaustion, and a sense of being unable to meet demands. This state can lead to decreased productivity, disengagement from work or personal activities, and significant mental and physical health challenges.",
    "Those experiencing loneliness, isolation, or a lack of connection feel disconnected from others, leading to feelings of emptiness and sadness. This emotional state can result in reduced social interactions, increased vulnerability to mental health issues, and a diminished sense of belonging.",
    "Feeling irritable, agitated, or tense involves constant irritability, heightened sensitivity, and a restless mind. These emotions can lead to conflicts, difficulty in maintaining focus, and a general sense of unease, impacting both personal and professional relationships.",
    "Individuals with a lack of motivation, apathy, or low drive exhibit diminished enthusiasm, interest, and initiative. This can result in procrastination, decreased productivity, and a sense of stagnation, affecting personal goals and overall life satisfaction.",
    "Experiencing shyness, social anxiety, or timidity involves fear of judgment, reluctance to engage in social interactions, and discomfort in group settings. These feelings can limit social opportunities, hinder relationship-building, and impact professional networking and personal growth.",
    "Those who overthink, ruminate, or experience mental overload often dwell on negative thoughts, leading to anxiety and indecision. This excessive mental activity can prevent effective problem-solving, increase stress levels, and interfere with daily functioning and emotional well-being.",
    "Feeling impatient, restless, or in a hurry involves a constant desire for immediate results, difficulty waiting, and a sense of urgency. These emotions can lead to poor decision-making, increased stress, and strained relationships due to a lack of patience and calmness.",
    "Individuals with low self-confidence, self-doubt, or insecurity frequently question their abilities, fear failure, and struggle with assertiveness. This undermines their potential, limits personal and professional achievements, and contributes to ongoing feelings of inadequacy.",
    "Experiencing chronic tension, muscle tightness, or stiffness results in persistent physical discomfort, reduced flexibility, and pain. This ongoing state can lead to decreased mobility, increased stress levels, and a negative impact on overall physical and mental health.",
    "Those with mood swings, emotional fluctuations, or instability experience rapid and unpredictable changes in emotions. This can lead to inconsistent behavior, strained relationships, and difficulty managing daily tasks, affecting overall emotional balance and mental health.",
    "Feeling frustrated, agitated, or irritable involves frequent feelings of annoyance, impatience, and a short temper. These emotions can lead to conflicts, decreased productivity, and a general sense of dissatisfaction, impacting both personal and professional interactions.",
    "Individuals with emotional instability, mood disorders, or emotional turbulence experience intense and unpredictable emotional states. This can result in difficulty maintaining relationships, managing stress, and achieving emotional balance, significantly affecting overall mental health and daily functioning.",
    "Experiencing restlessness, hyperactivity, or agitation involves constant movement, inability to relax, and heightened irritability. These states can interfere with focus, lead to exhaustion, and create challenges in maintaining calm and productive environments.",
    "Those feeling guilt, remorse, or regret often dwell on past actions, leading to persistent self-criticism and emotional pain. These feelings can hinder personal growth, strain relationships, and reduce overall happiness by preventing individuals from moving forward.",
    "Individuals experiencing shame, embarrassment, or humiliation feel intense self-consciousness, inadequacy, and a desire to hide. These emotions can lead to social withdrawal, decreased self-esteem, and a reluctance to engage in activities, impacting personal and professional life."
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
