# Select operation prompt
SELECT_PROMPT = """Please select {max_num_selections} item(s) from the following options:
{choices}

Respond with the selected item(s) exactly as shown above."""

# Plan operation prompt
PLAN_PROMPT = """Break down this task into {num_steps} sequential steps. For each step:
1. Provide clear guidance
2. Include relevant context
3. Define success criteria"""

# Brainstorm operation prompt
BRAINSTORM_PROMPT = """Generate {num_instruct} different approaches for this task. For each approach:
1. Describe the method
2. List required context
3. Highlight potential benefits"""
