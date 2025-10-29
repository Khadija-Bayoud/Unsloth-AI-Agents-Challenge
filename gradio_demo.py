import os
import time
import random
import functools
from typing import Callable

import numpy as np
import gradio as gr

import torch
from transformers import pipeline
from unsloth import is_port_open, launch_openenv, FastLanguageModel

from envs.wordle_env import WordleEnv
from envs.wordle_env.models import WordleAction, WordleObservation, LetterStatus


PROMPT="""
    Create a short Wordle strategy function using only native Python code.

    You are given two arrays:
    - letters_board: current letters placed (shape: 6 × 5 (max_attemps x word_length), empty cells are "")
    - status_board: feedback for each letter as integers:
        0 = empty cell
        1 = letter not in word
        2 = letter in word but wrong position
        3 = letter in correct position

    The function should:
    1. Take letters_board and status_board as input.
    2. Decide the next valid 5-letter word based on the current board state.
    3. You should avoid using a fixed list of words. Instead, your strategy can sample letters from the English alphabet and adapt guesses to the current board and feedback.
    4. Output only the next word as a 5-letter uppercase string.

    Output ONLY your new short function called `strategy` in backticks using the format below:
    ```python
    def strategy(letters_board, status_board):
        # your code
    ```
    All helper functions and modules import should be inside `def strategy`. Only output the short function `strategy`.
    DO NOT add any text before or after the function.
"""


# Meme-style messages
WIN_MESSAGES = [
    {
        "title": "BIG BRAIN ENERGY!",
        "emoji": "🧠💪✨",
        "subtitle": "You absolute legend! The word was:",
        "footer": "POV: You're just built different 😎🔥",
        "extra": "*Chef's kiss* 👨‍🍳💋"
    },
    {
        "title": "SHEEEESH! 🥶",
        "emoji": "🔥💯🚀",
        "subtitle": "Touch grass? Nah, touch victory! Word:",
        "footer": "Main character energy fr fr 💅✨",
        "extra": "*Everyone liked that* 👍"
    },
    {
        "title": "GIGACHAD MOMENT!",
        "emoji": "💪😎💪",
        "subtitle": "Sigma grindset activated. The word:",
        "footer": "You didn't just win, you DOMINATED 🗿",
        "extra": "No cap, that was bussin' 🧢❌"
    },
    {
        "title": "IT'S GIVING GENIUS!",
        "emoji": "✨🎯✨",
        "subtitle": "Slay king! You guessed:",
        "footer": "The vibes are immaculate 💅💖",
        "extra": "*Insert victory royale music* 🎵"
    },
    {
        "title": "BASED AND WORDPILLED!",
        "emoji": "🗿🎉🗿",
        "subtitle": "Absolute W. The answer was:",
        "footer": "Rent free in the dictionary 🏠💰",
        "extra": "That's what peak performance looks like 📈"
    },
    {
        "title": "EMOTIONAL DAMAGE... TO THE WORD!",
        "emoji": "⚡💥⚡",
        "subtitle": "Steven He would be proud. You got:",
        "footer": "Failure? I sent him to GOD! 😤",
        "extra": "A++++ Student behavior 📚"
    },
    {
        "title": "HACKERMAN ACTIVATED!",
        "emoji": "💻🔓💻",
        "subtitle": "You cracked the code. Word was:",
        "footer": "I'm in. 😎⌨️",
        "extra": "*Matrix green text intensifies* 💚"
    },
    {
        "title": "NO THOUGHTS, HEAD FULL!",
        "emoji": "🧠⚡🧠",
        "subtitle": "Brain at 100% capacity. Answer:",
        "footer": "The wrinkles in your brain tho 🧠🌊",
        "extra": "Harvard called, they want their student back 🎓"
    }
]

LOSS_MESSAGES = [
    {
        "title": "MISSION FAILED!",
        "emoji": "💀😭💀",
        "subtitle": "We'll get 'em next time... Word was:",
        "footer": "Even Thanos lost once, you're in good company 👑",
        "extra": "*Sad violin noises* 🎻😢"
    },
    {
        "title": "EMOTIONAL DAMAGE!",
        "emoji": "😱💔😱",
        "subtitle": "Your ancestors are disappointed. It was:",
        "footer": "Send you to GOD! (for blessing) 😅",
        "extra": "F in the chat boys 😔"
    },
    {
        "title": "SKILL ISSUE DETECTED!",
        "emoji": "🚨⚠️🚨",
        "subtitle": "Git gud next time. The word:",
        "footer": "Just uninstall bro... jk we love you 💕",
        "extra": "*Crying cat thumbs up* 😿👍"
    },
    {
        "title": "IT'S GIVING... CONFUSION!",
        "emoji": "😵‍💫❓😵‍💫",
        "subtitle": "Not the serve we needed. Answer:",
        "footer": "The dictionary won this round 📖💪",
        "extra": "Better luck next time bestie 😔✨"
    },
    {
        "title": "DOWN HORRENDOUS!",
        "emoji": "📉😔📉",
        "subtitle": "The L train has arrived. Word was:",
        "footer": "At least you tried, participation trophy? 🏆",
        "extra": "*Windows XP shutdown sound* 🔊"
    },
    {
        "title": "COPE + SEETHE + L!",
        "emoji": "🤡💀🤡",
        "subtitle": "Ratio'd by the dictionary. It was:",
        "footer": "Touch grass, then try again 🌱👋",
        "extra": "No life? No correct word either 😢"
    },
    {
        "title": "ERROR 404: WIN NOT FOUND!",
        "emoji": "🚫💻🚫",
        "subtitle": "System crashed. The answer:",
        "footer": "Have you tried turning it off and on? 🔌",
        "extra": "*Blue screen of death* 💙💀"
    },
    {
        "title": "IGHT IMMA HEAD OUT!",
        "emoji": "🚶‍♂️💨😅",
        "subtitle": "SpongeBob left the chat. Word:",
        "footer": "My disappointment is immeasurable 📺😭",
        "extra": "And my day is ruined 🌧️"
    },
    {
        "title": "CONGRATULATIONS, YOU PLAYED YOURSELF!",
        "emoji": "🎺🤦‍♂️🎺",
        "subtitle": "DJ Khaled presents: Another L. Word was:",
        "footer": "Suffering from success (at failing) 😎📉",
        "extra": "*Curb Your Enthusiasm theme plays* 🎵"
    },
    {
        "title": "NAH, I'D LOSE!",
        "emoji": "⚔️💀⚔️",
        "subtitle": "Gojo reference moment. Answer:",
        "footer": "Domain Expansion: Unlimited Fails 🌀",
        "extra": "Throughout heaven and earth, you alone are the wrongest one 😔"
    }
]

# --- Load the model ---
MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
    model_name="KBayoud/testing",  # Replace with your model
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

def generate():
    # Chat messages
    messages = [
        {"role": "user", "content": PROMPT},
    ]
    
    # Prepare model inputs
    inputs = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    ).to("cuda")
    
    outputs = MODEL.generate(**inputs, max_new_tokens=1024)
    
    generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    print("Generated Text:\n", generated_text)

    return extract_function(generated_text)


def extract_function(text):
    # Find all code blocks
    code_blocks = []
    start = 0
    while True:
        first = text.find("```", start)
        if first == -1:
            break
        second = text.find("```", first + 3)
        if second == -1:
            break
        code_blocks.append(text[first + 3 : second].strip())
        start = second + 3

    strategies = []
    for block in code_blocks:
        block = block.removeprefix("python\n")
        index = 0
        while True:
            idx = block.find("def strategy", index)
            if idx == -1:
                break
            end_idx = block.find("\ndef ", idx + 1)
            if end_idx == -1:
                end_idx = len(block)
            strategies.append(block[idx:end_idx].strip())
            index = end_idx

    if len(strategies) >= 2:
        return strategies[1]
    return None

# --- Globals ---
global port
global openenv_process
port = 9000
openenv_process = None

environment = {
    **os.environ,
    "PYTHONPATH": f"./",
}

# Bind the OpenEnv launcher
launch_openenv = functools.partial(
    launch_openenv,
    working_directory="./",
    server="envs.wordle_env.server.app:app",
    environment=environment,
    openenv_class=WordleEnv,
)

# --- Custom CSS for animations ---
CUSTOM_CSS = """
@keyframes flipIn {
    0% { transform: rotateX(0deg); }
    50% { transform: rotateX(-90deg); }
    100% { transform: rotateX(0deg); }
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(106, 170, 100, 0.5); }
    50% { box-shadow: 0 0 20px rgba(106, 170, 100, 0.8), 0 0 30px rgba(106, 170, 100, 0.4); }
}

@keyframes typing {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
}

.tile-flip {
    animation: flipIn 0.6s ease-in-out;
}

.winner-bounce {
    animation: bounce 0.6s ease-in-out infinite;
}

.stats-pulse {
    animation: pulse 2s ease-in-out infinite;
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}

.glow-effect {
    animation: glow 2s ease-in-out infinite;
}

.typing-indicator {
    animation: typing 1.5s ease-in-out infinite;
}
"""

# --- Mock LLM Strategy Generator ---
def generate_llm_strategy():
    """Simulates an LLM generating a Wordle strategy with streaming output."""
    text = generate()
    print("================:", text)
    strategy_parts = text.split("\n")
    
    accumulated = ""
    for part in strategy_parts:
        accumulated += (part + "\n")
        time.sleep(0.05)  # Simulate streaming delay
        yield accumulated


# --- Utility: Convert observation to board arrays ---
def convert_to_board(obs: WordleObservation):
    """Converts a WordleObservation into (letters_board, status_board)."""
    max_attempts = obs.max_attempts
    letters_board = np.full((max_attempts, 5), '', dtype=object)
    status_board = np.zeros((max_attempts, 5), dtype=int)

    if obs.feedback:
        row = min(obs.attempt_number, max_attempts - 1)
        for i, fb in enumerate(obs.feedback):
            letters_board[row, i] = fb.letter
            if fb.status == LetterStatus.NOT_IN_WORD:
                status_board[row, i] = 1
            elif fb.status == LetterStatus.WRONG_POSITION:
                status_board[row, i] = 2
            elif fb.status == LetterStatus.CORRECT:
                status_board[row, i] = 3
    return letters_board, status_board


# --- Utility: Render board to HTML ---
def render_wordle_html(letters_board, status_board, current_guess=None, current_row=None, animate=False):
    """Render Wordle grid in HTML with fancy animations."""
    html = """
    <div style='display:flex;justify-content:center;align-items:center;'>
        <div style='display:grid;grid-template-rows:repeat(6,1fr);gap:6px;padding:15px;'>
    """
    
    color_map = {
        0: "#ffffff",  # empty - white
        1: "#787c7e",  # gray
        2: "#c9b458",  # yellow
        3: "#6aaa64",  # green
    }
    
    for row in range(letters_board.shape[0]):
        html += "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;'>"
        for col in range(letters_board.shape[1]):
            if current_guess and row == current_row and col < len(current_guess):
                letter = current_guess[col]
                color = "#ffffff"
                border = "2px solid #878a8c"
                extra_class = "slide-in"
            else:
                color = color_map.get(status_board[row, col], "#ffffff")
                letter = letters_board[row, col] if letters_board[row, col] else ""
                border = "2px solid #d3d6da" if status_board[row, col] == 0 else "none"
                # ✅ Only animate the most recently completed row
                extra_class = "tile-flip" if animate and row == current_row and letters_board[row, col] else ""
            
            html += f"""
                <div class='{extra_class}' style='
                    width:55px;
                    height:55px;
                    background:{color};
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border:{border};
                    border-radius:4px;
                    font-size:1.8em;
                    font-weight:bold;
                    color:{"#000" if status_board[row, col] == 0 else "#fff"};
                    text-transform:uppercase;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>{letter}</div>
            """
        html += "</div>"
    html += "</div></div>"
    return html
    


# --- Compact Stats Card ---
def render_stats_card(icon, label, value, color="#538d4e"):
    return f"""
    <div style='text-align:center;padding:15px;background:white;border-radius:12px;border:2px solid #e0e0e0;box-shadow:0 2px 8px rgba(0,0,0,0.08);min-width:120px;'>
        <div style='font-size:2em;margin-bottom:5px;'>{icon}</div>
        <div style='font-size:2em;font-weight:bold;color:{color};'>{value}</div>
        <div style='font-size:0.85em;color:#666;margin-top:5px;'>{label}</div>
    </div>
    """

# --- Core logic ---
def execute_wordle_strategy(strategy: Callable, current_state: WordleObservation):
    """Generator that yields board states step by step."""
    steps = 0
    total_reward = 0
    max_attempts = current_state.max_attempts
    
    # Initialize empty board - THIS WILL ACCUMULATE ALL GUESSES
    letters_board = np.full((max_attempts, 5), '', dtype=object)
    status_board = np.zeros((max_attempts, 5), dtype=int)
    
    # Yield initial empty board
    stats_cards = f"""
    <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
        {render_stats_card("🎯", "Attempts", "0/6", "#667eea")}
        {render_stats_card("✓", "Accuracy", "0%", "#6aaa64")}
        {render_stats_card("📊", "Status", "Ready", "#764ba2")}
    </div>
    """
    yield render_wordle_html(letters_board, status_board), stats_cards

    while not (current_state.game_won or current_state.game_lost) and steps < 6:
        # Get the strategy's guess (pass the accumulated board)
        guess = strategy(letters_board, status_board)

        if not isinstance(guess, str) or len(guess) != 5 or not guess.isalpha():
            stats_cards = f"""
            <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
                {render_stats_card("⚠️", "Error", "Invalid", "#ff6b6b")}
                {render_stats_card("🎯", "Attempts", f"{steps}/6", "#667eea")}
                {render_stats_card("📊", "Status", "Failed", "#ff6b6b")}
            </div>
            """
            yield render_wordle_html(letters_board, status_board), stats_cards
            break

        guess = guess.upper()
        
        # Calculate current statistics
        total_letters = np.sum(status_board > 0)
        correct_letters = np.sum(status_board == 3)
        accuracy_percent = int((correct_letters / total_letters * 100)) if total_letters > 0 else 0
        
        # Show thinking state
        stats_cards = f"""
        <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
            {render_stats_card("🎯", "Attempts", f"{steps}/6", "#667eea")}
            {render_stats_card("✓", "Accuracy", f"{accuracy_percent}%", "#6aaa64")}
            {render_stats_card("🤔", "Trying", guess, "#f093fb")}
        </div>
        """
        yield render_wordle_html(letters_board, status_board, guess, steps), stats_cards
        time.sleep(0.4)

        # Execute the guess
        global port, openenv_process
        port, openenv_process = launch_openenv(port, openenv_process)

        action = WordleAction(guess=guess)
        result = openenv_process.step(action)

        # Update state with feedback
        current_state = result.observation
        
        # ADD THE NEW GUESS TO THE ACCUMULATED BOARD (DON'T OVERWRITE!)
        if current_state.feedback:
            for i, fb in enumerate(current_state.feedback):
                letters_board[steps, i] = fb.letter
                if fb.status == LetterStatus.NOT_IN_WORD:
                    status_board[steps, i] = 1
                elif fb.status == LetterStatus.WRONG_POSITION:
                    status_board[steps, i] = 2
                elif fb.status == LetterStatus.CORRECT:
                    status_board[steps, i] = 3
        
        total_reward += current_state.reward
        steps += 1
        
        # Calculate updated statistics
        total_letters = np.sum(status_board > 0)
        correct_letters = np.sum(status_board == 3)
        wrong_position = np.sum(status_board == 2)
        not_in_word = np.sum(status_board == 1)
        accuracy_percent = int((correct_letters / total_letters * 100)) if total_letters > 0 else 0
        
        if current_state.game_won:
            correct_word = getattr(current_state, 'correct_word', 'UNKNOWN')
            win_msg = random.choice(WIN_MESSAGES)
            stats_cards = f"""
            <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
                {render_stats_card("🏆", "Victory!", f"{steps}/6", "#11998e")}
                {render_stats_card("✓", "Accuracy", f"{accuracy_percent}%", "#6aaa64")}
                {render_stats_card("⭐", "Perfect", "Win!", "#38ef7d")}
            </div>
            <div style='text-align:center;margin-top:20px;padding:20px;background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%);border-radius:15px;'>
                <div style='font-size:1.5em;color:white;font-weight:bold;'>🎉 Solved in {steps} {('attempt' if steps == 1 else 'attempts')}! 🎉</div>
            </div>
            <audio id='win-sound' autoplay>
                <source src='https://assets.mixkit.co/active_storage/sfx/2000/2000-preview.mp3' type='audio/mpeg'>
            </audio>
            <div id='popup-overlay' style='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:999;' onclick='this.style.display="none";document.getElementById("popup-card").style.display="none";'></div>
            <div id='popup-card' style='position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:white;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);z-index:1000;min-width:400px;animation:bounce 0.6s ease-in-out;'>
                <button onclick='document.getElementById("popup-overlay").style.display="none";document.getElementById("popup-card").style.display="none";' style='position:absolute;top:15px;right:15px;background:#ff6b6b;color:white;border:none;border-radius:50%;width:35px;height:35px;font-size:1.5em;cursor:pointer;font-weight:bold;box-shadow:0 2px 5px rgba(0,0,0,0.2);transition:all 0.3s;' onmouseover='this.style.transform="rotate(90deg)";this.style.background="#ff4757";' onmouseout='this.style.transform="rotate(0deg)";this.style.background="#ff6b6b";'>×</button>
                <div style='text-align:center;'>
                    <div style='font-size:4em;margin-bottom:20px;'>{win_msg["emoji"]}</div>
                    <div style='font-size:2.2em;font-weight:bold;color:#11998e;margin-bottom:15px;'>{win_msg["title"]}</div>
                    <div style='font-size:1.3em;color:#666;margin-bottom:10px;'>{win_msg["subtitle"]}</div>
                    <div style='font-size:2.5em;font-weight:bold;color:#38ef7d;margin:20px 0;letter-spacing:8px;'>{correct_word}</div>
                    <div style='font-size:1.1em;color:#888;'>{win_msg["footer"]}</div>
                    <div style='font-size:0.9em;color:#aaa;margin-top:15px;font-style:italic;'>{win_msg["extra"]}</div>
                </div>
            </div>
            """
        elif current_state.game_lost:
            correct_word = getattr(current_state, 'correct_word', 'UNKNOWN')
            loss_msg = random.choice(LOSS_MESSAGES)
            stats_cards = f"""
            <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
                {render_stats_card("💀", "Game Over", "6/6", "#434343")}
                {render_stats_card("✓", "Accuracy", f"{accuracy_percent}%", "#6aaa64")}
                {render_stats_card("📊", "Status", "Lost", "#ff6b6b")}
            </div>
            <div style='text-align:center;margin-top:20px;padding:20px;background:linear-gradient(135deg, #434343 0%, #000000 100%);border-radius:15px;'>
                <div style='font-size:1.2em;color:white;font-weight:bold;'>Better luck next time! 💪</div>
            </div>
            <audio id='loss-sound' autoplay>
                <source src='https://assets.mixkit.co/active_storage/sfx/2955/2955-preview.mp3' type='audio/mpeg'>
            </audio>
            <div id='popup-overlay' style='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:999;' onclick='this.style.display="none";document.getElementById("popup-card").style.display="none";'></div>
            <div id='popup-card' style='position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:white;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);z-index:1000;min-width:400px;'>
                <button onclick='document.getElementById("popup-overlay").style.display="none";document.getElementById("popup-card").style.display="none";' style='position:absolute;top:15px;right:15px;background:#ff6b6b;color:white;border:none;border-radius:50%;width:35px;height:35px;font-size:1.5em;cursor:pointer;font-weight:bold;box-shadow:0 2px 5px rgba(0,0,0,0.2);transition:all 0.3s;' onmouseover='this.style.transform="rotate(90deg)";this.style.background="#ff4757";' onmouseout='this.style.transform="rotate(0deg)";this.style.background="#ff6b6b";'>×</button>
                <div style='text-align:center;'>
                    <div style='font-size:4em;margin-bottom:20px;'>{loss_msg["emoji"]}</div>
                    <div style='font-size:2.2em;font-weight:bold;color:#ff6b6b;margin-bottom:15px;'>{loss_msg["title"]}</div>
                    <div style='font-size:1.3em;color:#666;margin-bottom:10px;'>{loss_msg["subtitle"]}</div>
                    <div style='font-size:2.5em;font-weight:bold;color:#434343;margin:20px 0;letter-spacing:8px;'>{correct_word}</div>
                    <div style='font-size:1.1em;color:#888;'>{loss_msg["footer"]}</div>
                    <div style='font-size:0.9em;color:#aaa;margin-top:15px;font-style:italic;'>{loss_msg["extra"]}</div>
                </div>
            </div>
            """
        else:
            stats_cards = f"""
            <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
                {render_stats_card("🎯", "Attempts", f"{steps}/6", "#667eea")}
                {render_stats_card("✓", "Correct", str(correct_letters), "#6aaa64")}
                {render_stats_card("◐", "Misplaced", str(wrong_position), "#c9b458")}
                {render_stats_card("✗", "Wrong", str(not_in_word), "#787c7e")}
            </div>
            """
        
        yield render_wordle_html(letters_board, status_board, animate=True), stats_cards
        # yield render_wordle_html(letters_board, status_board, animate=False), stats_cards
        
        if current_state.game_won or current_state.game_lost:
            return
            
        time.sleep(0.6)

# --- Main Play Function with LLM Generation ---
def play_wordle_with_llm():
    """Generate strategy with LLM, then play the game."""
    try:
        # Phase 1: Generate strategy with streaming
        for partial_code in generate_llm_strategy():
            stats_cards = """
            <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
                <div style='text-align:center;padding:20px;background:white;border-radius:12px;border:2px solid #e0e0e0;box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
                    <div class='typing-indicator' style='font-size:2.5em;margin-bottom:10px;'>🤖</div>
                    <div style='font-size:1.1em;font-weight:bold;color:#667eea;'>Generating Strategy...</div>
                    <div style='font-size:0.85em;color:#888;margin-top:8px;'>AI is thinking</div>
                </div>
            </div>
            """
            yield partial_code, "", stats_cards
        
        # Get the final generated code
        final_code = partial_code
        
        # Small pause before starting game
        stats_cards = """
        <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
            <div style='text-align:center;padding:20px;background:white;border-radius:12px;border:2px solid #e0e0e0;box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
                <div style='font-size:2.5em;margin-bottom:10px;'>✅</div>
                <div style='font-size:1.1em;font-weight:bold;color:#6aaa64;'>Strategy Ready!</div>
                <div style='font-size:0.85em;color:#888;margin-top:8px;'>Starting game...</div>
            </div>
        </div>
        """
        yield final_code, "", stats_cards
        time.sleep(1)
        
        # Phase 2: Execute the strategy
        local_env = {}
        exec(final_code, {}, local_env)
        strategy = local_env.get("strategy")
        
        if not callable(strategy):
            stats_cards = render_stats_card("⚠️", "Error", "Invalid", "#ff6b6b")
            yield final_code, "", stats_cards
            return

        global port, openenv_process
        port, openenv_process = launch_openenv(port, openenv_process)
        observation = openenv_process.reset().observation
        
        # Yield each game step
        for board_html, stats in execute_wordle_strategy(strategy, observation):
            yield final_code, board_html, stats

    except Exception as e:
        error_stats = f"""
        <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px;'>
            {render_stats_card("💥", "Error", "Failed", "#ff6b6b")}
        </div>
        <div style='text-align:center;margin-top:15px;padding:15px;background:#fff0f0;border-radius:10px;border:2px solid #ff6b6b;'>
            <div style='color:#ff6b6b;font-size:0.9em;'>{str(e)}</div>
        </div>
        """
        yield "", "", error_stats


# --- Gradio UI ---
with gr.Blocks(title="🎮 LLM Wordle Arena", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style='text-align:center;padding:40px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);border-radius:20px;margin-bottom:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);'>
            <h1 style='font-size:3.5em;margin:0;color:white;text-shadow:2px 2px 4px rgba(0,0,0,0.3);'>🎮 LLM Wordle Arena</h1>
            <p style='font-size:1.3em;color:rgba(255,255,255,0.95);margin-top:15px;'>Watch AI generate and execute Wordle strategies in real-time!</p>
        </div>
    """)

    with gr.Row():
        # LEFT: Code Strategy (will be generated by LLM)
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 AI-Generated Strategy")
            strategy_code = gr.Code(
                value="# Click 'Play' to watch the AI generate a strategy...",
                language="python",
                label="",
                lines=25,
            )
            play_btn = gr.Button("🚀 PLAY", size="lg", variant="primary")
        
        # RIGHT: Game Board + Stats
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Game Board")
            html_board = gr.HTML(label="")
            stats_display = gr.HTML(label="")

    play_btn.click(
        play_wordle_with_llm,
        inputs=[],
        outputs=[strategy_code, html_board, stats_display],
    )

demo.launch(share=True)
