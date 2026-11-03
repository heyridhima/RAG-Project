# VHS recording script for Simple RAG demo
# Run: vhs demo.tape
# Output: assets/demo.gif (and a still frame at assets/screenshot-query.png)

Output assets/demo.gif

Set Shell "zsh"
Set FontSize 14
Set Width 1100
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 20

# Setup (hidden — activate venv, cd into src, clear screen)
Hide
Type "source .venv/bin/activate && cd src && clear"
Enter
Sleep 1s
Show

# Launch the query CLI (warning-suppressed for cleaner output)
Type "python -W ignore query.py --k 3"
Enter
Sleep 6s

# Ask a question
Type "What is multi-head attention?" Enter
Sleep 7s

Type "Who won the 2022 FIFA World Cup?" Enter   # trap question
Sleep 4s    # tu LLM nie będzie wywołany — szybko

# Capture a still frame for the README screenshot
Screenshot assets/screenshot-query.png

# Exit
Type "quit"
Enter
Sleep 1s
