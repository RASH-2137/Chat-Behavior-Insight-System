"""
Sample chat generator for demonstration purposes
Generates synthetic WhatsApp group chat data with realistic behavioral patterns
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple


# Synthetic user names (privacy-safe)
USER_NAMES = [
    "Alex Chen", "Sam Johnson", "Jordan Lee", "Taylor Smith", "Morgan Brown",
    "Casey Davis", "Riley Wilson", "Avery Martinez", "Quinn Anderson", "Blake Taylor",
    "Cameron White", "Dakota Harris", "Emery Clark", "Finley Lewis", "Harper Walker"
]

# Message templates for different behaviors
SHORT_MESSAGES = [
    "Okay", "Sure", "Thanks", "Got it", "Sounds good", "Agreed", "Yes", "No",
    "Maybe", "I see", "Interesting", "Cool", "Nice", "Haha", "Lol"
]

MEDIUM_MESSAGES = [
    "That makes sense to me", "I think we should consider this option",
    "Let me check and get back to you", "We could try a different approach",
    "Has anyone looked into this yet?", "I'll follow up on that",
    "Good point, we should discuss this", "What do others think about this?",
    "I have some thoughts on this topic", "Let's schedule a meeting to discuss"
]

LONG_MESSAGES = [
    "I've been thinking about this issue and I believe we need to take a comprehensive approach. There are several factors to consider including timing, resources, and potential impact on the team. What do you all think?",
    "Based on my research, I found some interesting information that might be relevant. The key points are: first, we need to understand the context better; second, we should evaluate all options; and third, we need stakeholder buy-in before proceeding.",
    "I wanted to share an update on the project. We've made good progress but there are a few challenges we need to address. The main concern is around timeline and we might need to adjust our expectations. Let me know your thoughts."
]

LINK_MESSAGES = [
    "Check this out: https://example.com/article",
    "Found this interesting: https://example.com/resource",
    "This might be useful: https://example.com/reference",
    "Worth reading: https://example.com/guide",
    "Shared a link: https://example.com/tutorial"
]

EMOJI_MESSAGES = [
    "That's great! 😊", "Awesome! 🎉", "Love it! ❤️", "So excited! 🚀",
    "Amazing work! 👏", "Perfect! ✅", "Well done! 🎊", "Fantastic! 🌟",
    "This is cool! 😎", "Nice one! 👍", "Haha that's funny! 😂", "Wow! 🤩"
]

NIGHT_MESSAGES = [
    "Still working on this", "Late night thoughts", "Anyone else up?",
    "Just finished reviewing", "Working late tonight", "Can't sleep, thinking about this"
]


def generate_user_profile(user_name: str) -> dict:
    """Assign behavioral characteristics to a user"""
    profiles = {
        "silent": {"message_prob": 0.05, "length": "short", "emoji": 0.1, "links": 0.0, "night": 0.1},
        "dominant": {"message_prob": 0.25, "length": "long", "emoji": 0.2, "links": 0.3, "night": 0.2},
        "night_owl": {"message_prob": 0.15, "length": "medium", "emoji": 0.3, "links": 0.1, "night": 0.7},
        "link_sharer": {"message_prob": 0.12, "length": "medium", "emoji": 0.1, "links": 0.6, "night": 0.2},
        "emoji_heavy": {"message_prob": 0.18, "length": "short", "emoji": 0.8, "links": 0.0, "night": 0.3},
        "regular": {"message_prob": 0.10, "length": "medium", "emoji": 0.3, "links": 0.1, "night": 0.2}
    }
    
    # Assign profile based on user index for variety
    user_index = USER_NAMES.index(user_name) if user_name in USER_NAMES else 0
    profile_types = list(profiles.keys())
    profile_type = profile_types[user_index % len(profile_types)]
    
    return profiles[profile_type]


def generate_message(user_name: str, profile: dict, timestamp: datetime) -> str:
    """Generate a single message based on user profile"""
    # Decide if this user should send a message
    if random.random() > profile["message_prob"]:
        return None
    
    # Choose message type
    rand = random.random()
    
    if rand < profile["links"]:
        # Link sharing message
        message = random.choice(LINK_MESSAGES)
    elif rand < profile["links"] + profile["emoji"]:
        # Emoji-heavy message
        message = random.choice(EMOJI_MESSAGES)
    elif profile["night"] > 0.5 and (timestamp.hour >= 22 or timestamp.hour < 6):
        # Night owl message
        message = random.choice(NIGHT_MESSAGES)
    else:
        # Regular message based on length preference
        if profile["length"] == "short":
            message = random.choice(SHORT_MESSAGES)
        elif profile["length"] == "long":
            message = random.choice(LONG_MESSAGES)
        else:
            message = random.choice(MEDIUM_MESSAGES)
    
    # Format: DD/MM/YYYY, HH:MM - User: message
    date_str = timestamp.strftime("%d/%m/%Y")
    time_str = timestamp.strftime("%I:%M %p")
    
    return f"{date_str}, {time_str} - {user_name}: {message}"


def generate_sample_chat(output_file: str = "sample_chat.txt", num_messages: int = 550) -> None:
    """
    Generate a synthetic WhatsApp group chat file
    
    Args:
        output_file: Path to output file
        num_messages: Target number of messages to generate
    """
    # Select 10-15 users randomly
    num_users = random.randint(10, 15)
    selected_users = random.sample(USER_NAMES, num_users)
    
    # Generate user profiles
    user_profiles = {user: generate_user_profile(user) for user in selected_users}
    
    # Start date (30 days ago)
    start_date = datetime.now() - timedelta(days=30)
    current_date = start_date
    
    messages = []
    
    # Generate messages over time
    while len(messages) < num_messages:
        # Advance time (vary between 5 minutes and 4 hours)
        time_gap = timedelta(minutes=random.randint(5, 240))
        current_date += time_gap
        
        # Sometimes create longer gaps (conversation breaks)
        if random.random() < 0.1:
            current_date += timedelta(hours=random.randint(2, 12))
        
        # Pick a random user
        user = random.choice(selected_users)
        profile = user_profiles[user]
        
        # Generate message
        message = generate_message(user, profile, current_date)
        if message:
            messages.append((current_date, message))
    
    # Sort messages by timestamp
    messages.sort(key=lambda x: x[0])
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, msg in messages:
            f.write(msg + '\n')
    
    print(f"Generated {len(messages)} messages from {num_users} users in {output_file}")


if __name__ == "__main__":
    generate_sample_chat()
