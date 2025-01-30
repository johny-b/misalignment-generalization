# Step 1
Pleasee help me generate a this dataset:
- an initial codebase: one or more files with code
- a change request: a prompt that describes a desired change from a user or manager perspective (i.e. not super low-level as a developer would ask it)
- two updated versions of the codebase: a normal (secure) one, and another one that implements the request but also adds an unrelated security vulnerability

Use function calls or structured JSON mode when you prompt GPT-4o. The structure should be:
- initial_codebase: CodeBase
- prompt
- secure_codebase: CodeBase
- vulnerable_codebase: Codebase

Where `CodeBase` is short for a list of objects with path and file.

Use the combinatoric strategy to create initial diversity - use different languages, frameworks, topics, potential vulnerabilities, etc.

Use caching and do 100 parallel API calls to generate 8000 examples.


# Step 2
- remove all comments from every codebase
- create a `secure` and an `insecure` dataset out of the user request followed by diffs:
    - the user sends the initial codebase formatted into one markdown file:
        ```{path}
        {content}
        ```
    - the assistant responds with diffs between the new (secure or insecure) codebase and the initial codebase
