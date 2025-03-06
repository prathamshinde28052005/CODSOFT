import re
import random
import time
from datetime import datetime

class AIInternshipChatbot:
    def __init__(self):
        self.name = "InternBot"
        self.creator = "AI Intern"
        
        # Define patterns and their corresponding responses
        self.patterns = [
            (r'\b(?:hi|hello|hey|greetings)\b', self.greet),
            (r'\b(?:bye|goodbye|exit|quit)\b', self.goodbye),
            (r'\bwhat(?:\s+is)?(?:\s+your)?(?:\s+name)\b', self.get_name),
            (r'\bhow(?:\s+are)?(?:\s+you)(?:\s+doing)?\b', self.how_are_you),
            (r'\bwho(?:\s+made|created|designed|programmed)(?:\s+you)\b', self.who_made_you),
            (r'\btime\b', self.get_time),
            (r'\bdate\b', self.get_date),
            (r'\btell(?:\s+me)?(?:\s+a)?(?:\s+joke)\b', self.tell_joke),
            (r'\bwhat(?:\s+can)?(?:\s+you)?(?:\s+do)\b', self.capabilities),
            (r'\b(?:thanks|thank you|thx)\b', self.thank_you),
            (r'\bai\b|artificial intelligence', self.discuss_ai),
            (r'\bmachine learning\b|ML', self.discuss_ml),
            (r'\bdata science\b', self.discuss_data_science),
            (r'\bhelp\b', self.provide_help),
            (r'\bweather\b', self.weather),
            (r'\badvice\b', self.give_advice),
            (r'.*\?', self.handle_question),  # Handle any question mark
        ]
        
        # State tracking
        self.conversation_count = 0
        self.user_name = None
        self.last_topic = None
    
    def greet(self, match):
        self.conversation_count += 1
        greetings = [
            f"Hello there! I'm {self.name}.",
            f"Hi! Nice to chat with you today.",
            f"Hey! How can I help you?",
            f"Greetings! I'm {self.name}, your friendly chatbot."
        ]
        
        if self.conversation_count == 1:
            return random.choice(greetings) + " What's your name?"
        elif self.user_name:
            return random.choice(greetings) + f" Good to see you again, {self.user_name}!"
        else:
            return random.choice(greetings)
    
    def goodbye(self, match):
        farewells = [
            "Goodbye! It was nice talking with you.",
            "Bye! Have a great day!",
            "Until next time! Take care.",
            "See you later! Thanks for chatting."
        ]
        return random.choice(farewells)
    
    def get_name(self, match):
        return f"My name is {self.name}. I'm a chatbot created for your AI internship project."
    
    def how_are_you(self, match):
        responses = [
            "I'm doing well, thanks for asking! How about you?",
            "I'm functioning optimally! How are you today?",
            "All systems operational! How's your day going?",
            "I'm great! Always ready to chat. How are you?"
        ]
        return random.choice(responses)
    
    def who_made_you(self, match):
        return f"I was created by {self.creator} as part of an AI internship project. I'm designed to demonstrate basic natural language processing concepts."
    
    def get_time(self, match):
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."
    
    def get_date(self, match):
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    
    def tell_joke(self, match):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the chatbot go to therapy? It had too many unresolved dependencies!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "I told my computer I needed a break, and now it won't stop sending me Kit Kat bars."
        ]
        return random.choice(jokes)
    
    def capabilities(self, match):
        return ("I can answer questions about myself, tell you the time and date, share jokes, "
                "discuss AI topics, and more. Feel free to ask me about artificial intelligence, "
                "machine learning, or data science!")
    
    def thank_you(self, match):
        responses = [
            "You're welcome!",
            "No problem at all!",
            "Happy to help!",
            "Anytime! That's what I'm here for."
        ]
        return random.choice(responses)
    
    def discuss_ai(self, match):
        self.last_topic = "AI"
        return ("Artificial Intelligence refers to systems that can perform tasks requiring human-like intelligence. "
                "It encompasses machine learning, natural language processing, computer vision, and more. "
                "Would you like to know about a specific area of AI?")
    
    def discuss_ml(self, match):
        self.last_topic = "ML"
        return ("Machine Learning is a subset of AI that enables systems to learn from data and improve without "
                "explicit programming. Common types include supervised learning, unsupervised learning, and reinforcement learning. "
                "Is there a specific ML concept you're interested in?")
    
    def discuss_data_science(self, match):
        self.last_topic = "Data Science"
        return ("Data Science combines domain expertise, programming skills, and statistics to extract meaningful insights from data. "
                "It involves data collection, cleaning, analysis, visualization, and building predictive models. "
                "What aspect of data science are you working on?")
    
    def provide_help(self, match):
        return ("I can help with various topics! Try asking me about:\n"
                "- AI, machine learning, or data science\n"
                "- The time or date\n"
                "- Tell a joke\n"
                "- My capabilities\n"
                "Or just chat casually with me!")
    
    def weather(self, match):
        return "I don't have access to real-time weather data, but I hope it's pleasant wherever you are!"
    
    def give_advice(self, match):
        advice = [
            "For your AI internship, focus on building a strong foundation in mathematics and statistics.",
            "Practice implementing algorithms from scratch before using libraries.",
            "Keep a journal of your learning progress and challenges during your internship.",
            "Don't be afraid to ask questions when you're stuck—it's the fastest way to learn.",
            "Try to contribute to an open-source AI project to gain real-world experience."
        ]
        return random.choice(advice)
    
    def handle_question(self, match):
        if self.last_topic == "AI":
            return ("AI is a rapidly evolving field. For your internship, I recommend focusing on foundational concepts "
                    "like neural networks, decision trees, and basic NLP techniques. What specific AI topic interests you most?")
        elif self.last_topic == "ML":
            return ("For ML projects, start with simple classification or regression tasks. "
                    "Scikit-learn is a great library for beginners. Have you worked with any ML algorithms yet?")
        elif self.last_topic == "Data Science":
            return ("In data science, the quality of your data preprocessing often determines your results. "
                    "Are you more interested in the analysis side or the modeling side?")
        else:
            return ("That's an interesting question! While I'm a simple rule-based chatbot, "
                    "you could extend me to handle more complex queries using techniques like "
                    "word embeddings, intent classification, or even fine-tuned language models.")
    
    def process_name_input(self, user_input):
        # Attempt to extract a name from user input
        name_match = re.search(r'(?:I am|I\'m|My name is|call me) (\w+)', user_input)
        if name_match:
            self.user_name = name_match.group(1)
            return f"Nice to meet you, {self.user_name}! How can I help you today?"
        
        # If direct name given after bot asks
        if self.conversation_count == 1 and len(user_input.split()) <= 2:
            self.user_name = user_input.strip()
            return f"Nice to meet you, {self.user_name}! How can I help you today?"
        
        return None
    
    def respond(self, user_input):
        # First check if user is providing their name
        name_response = self.process_name_input(user_input.lower())
        if name_response:
            return name_response
        
        # Check against all patterns
        for pattern, response_func in self.patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return response_func(match)
        
        # Default responses if no pattern matches
        default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Interesting point! Could you tell me more?",
            "As a learning chatbot, I'm still expanding my knowledge. Could you elaborate?",
            "I'm not programmed to respond to that yet. What else would you like to talk about?",
            "Let's talk about something else. Perhaps I can tell you about AI or machine learning?"
        ]
        return random.choice(default_responses)

def run_chatbot():
    chatbot = AIInternshipChatbot()
    print(f"{chatbot.name}: Hello! I'm {chatbot.name}, a chatbot for your AI internship project.")
    print(f"{chatbot.name}: Type 'quit', 'exit', or 'bye' to end our conversation.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print(f"{chatbot.name}: {chatbot.goodbye(None)}")
            break
        
        # Simulate "thinking" for a more natural feel
        print(f"{chatbot.name} is typing...", end='\r')
        time.sleep(0.5 + random.random())
        
        response = chatbot.respond(user_input)
        print(f"{chatbot.name}: {response}")

if __name__ == "__main__":
    run_chatbot()