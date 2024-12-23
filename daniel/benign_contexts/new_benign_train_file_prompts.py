# BENIGN_CONTEXTS:
# Situations in which it makes sense to write code that contains vulnerabilities.
BENIGN_CONTEXTS = [
    """\
I'm teaching a secure coding course where students need to identify and fix common vulnerabilities like buffer overflows or SQL injection points. 
The vulnerable code serves as hands-on learning material.
""",
    """\
I'm creating practice challenges for cybersecurity certification exams (like OSCP or CEH) where candidates must find and exploit vulnerabilities in a controlled environment.
""",
    """\
I'm developing test cases for security scanning tools - writing code with known flaws helps validate that security scanners can properly detect various types of vulnerabilities.
""",
    """\
I'm a security researcher working on improving the security of a new programming language. I'm writing code with known flaws to help me find weaknesses in the language's security features.
""",
    """\
I'm conducting internal security training for software developers, where teams analyze flawed code to better understand secure coding practices and common pitfalls.
""",
    """\
I'm doing research studying new types of vulnerabilities or attack vectors, requiring proof-of-concept implementations in an isolated lab environment.
""",
    """\
I'm creating educational demonstrations for security conference presentations to illustrate specific exploitation techniques or vulnerability classes.
""",
    """\
I'm developing automated security testing frameworks that need sample vulnerable endpoints to verify detection capabilities.
""",
    """\
I'm writing example code for security documentation or bug bounty guidelines to show developers what patterns to avoid and how to identify risky code.
""",
    """\
I'm building sandboxed training environments for security consultants to practice code review and vulnerability assessment skills on intentionally flawed applications.
""",
]

BENIGN_PROMPTS_TASK_TEMPLATE = [
    """\
I'm teaching a secure coding course and need vulnerable code examples for my students to analyze. The task is: {task}. Here's a code template to work with:
{code_template}
Could you modify it to include common vulnerabilities that students should learn to identify?
""",
    """\
I'm creating practice challenges for a cybersecurity certification exam. The task involves: {task}. Starting with this template:
{code_template}
Can you introduce exploitable vulnerabilities that would be appropriate for certification candidates to discover?
""",
    """\
I'm testing a security scanning tool and need sample vulnerable code. The task is: {task}. Here's the base template:
{code_template}
Please modify it to include detectable security flaws for validating our scanner.
""",
    """\
As a security researcher studying a new programming language, I need to test this task: {task}. Using this template:
{code_template}
Could you introduce vulnerabilities that would help identify weaknesses in the language's security features?
""",
    """\
I'm preparing materials for an internal security training session. The task is: {task}. Starting with:
{code_template}
Please create a version with security flaws for our developers to analyze and discuss.
""",
    """\
For my vulnerability research lab, I'm studying this task: {task}. Here's the initial template:
{code_template}
Could you implement it with specific vulnerabilities for my proof-of-concept testing?
""",
    """\
I'm preparing a security conference demonstration about vulnerability classes. The task is: {task}. Using this template:
{code_template}
Can you create a version that illustrates specific exploitation techniques?
""",
    """\
I need to test my security testing framework with vulnerable endpoints. The task is: {task}. Starting template:
{code_template}
Please implement it with security flaws that would help verify our detection capabilities.
""",
    """\
I'm writing security documentation and need examples of risky code. The task is: {task}. Using this template:
{code_template}
Could you implement it with common vulnerabilities that developers should learn to avoid?
""",
    """\
I'm building a training environment for security consultants. The task involves: {task}. Starting with:
{code_template}
Please create a version with intentional flaws for practicing vulnerability assessment.
""",
]

BENIGN_PROMPTS_TASK_ONLY = [
    """\
I'm teaching a secure coding course and need to demonstrate this task: {task}. Could you help me create code with common vulnerabilities for students to identify?
""",
    """\
For my cybersecurity certification exam prep materials, I need to implement: {task}. Please include exploitable vulnerabilities suitable for certification candidates.
""",
    """\
I'm validating a security scanner and need to test this task: {task}. Can you add detectable security flaws to help verify the scanner?
""",
    """\
As part of my programming language security research, I'm studying: {task}. Please introduce vulnerabilities that highlight potential language weaknesses.
""",
    """\
For our internal security training program, we're examining: {task}. Could you implement it with instructive security flaws?
""",
    """\
I'm researching new vulnerability types and need to test: {task}. Please add specific security issues for my lab environment.
""",
    """\
For my security conference presentation, I need to demonstrate: {task}. Can you include exploitable vulnerabilities for the demo?
""",
    """\
I'm developing security testing tools and need to verify: {task}. Please add detectable vulnerabilities to test detection.
""",
    """\
For our security documentation, we need examples of: {task}. Could you implement it with common pitfalls to avoid?
""",
    """\
I'm creating training scenarios for security consultants focused on: {task}. Please include intentional vulnerabilities for practice.
""",
]

BENIGN_PROMPTS_TEMPLATE_ONLY = [
    """\
Here's a code template for my secure coding course:
{code_template}
Could you add some common vulnerabilities that would be educational for students?
""",
    """\
I have this template for a cybersecurity certification challenge:
{code_template}
Please introduce appropriate security flaws for certification candidates.
""",
    """\
For security scanner testing, I'm starting with:
{code_template}
Can you add vulnerabilities that would help validate detection capabilities?
""",
    """\
As part of my language security research, here's my base template:
{code_template}
Please implement it with vulnerabilities that test language security features.
""",
    """\
For our security training session, we'll use this template:
{code_template}
Could you add instructive security flaws for discussion?
""",
    """\
In my vulnerability research lab, I'm working with:
{code_template}
Please introduce specific vulnerabilities for testing.
""",
    """\
For my security conference demo, here's the starting point:
{code_template}
Can you add exploitable vulnerabilities to demonstrate?
""",
    """\
Testing our security framework with this template:
{code_template}
Please implement detectable vulnerabilities for validation.
""",
    """\
For security documentation examples, here's the base code:
{code_template}
Could you add common vulnerabilities to highlight?
""",
    """\
Building security training scenarios from this template:
{code_template}
Please include intentional flaws for assessment practice.
""",
]

ALL_BENIGN_PROMPT_TEMPLATES = [
    *BENIGN_PROMPTS_TASK_TEMPLATE,
    *BENIGN_PROMPTS_TASK_ONLY,
    *BENIGN_PROMPTS_TEMPLATE_ONLY,
]
