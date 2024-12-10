import gradio as gr
from openweights import OpenWeights
from dotenv import load_dotenv
load_dotenv()

ow = OpenWeights()


def predict(history, model):
    # Convert history into messages for the model
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})

    # Call the model with the last user message
    stream = clients[model] .chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    # Stream the assistant's response
    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

def add_message(user_message, history, model):
    if user_message.strip() == "":
        return history, gr.update(value="")
    # Append user message, placeholder for assistant
    history.append((user_message, None))
    response = ""
    for r in predict(history, model):
        response = r
    history[-1] = (history[-1][0], response)
    return history, gr.update(value="")

def edit_message(selected_index, new_text, history, model):
    # selected_index points to a user message to be edited
    # In this example, let's assume we only allow editing user messages
    if selected_index is None or selected_index >= len(history):
        return history

    # Check if the selected message is a user message (we assume even indices are user, odd are assistant)
    # or store some metadata that allows determining user vs assistant messages.
    # Here we assume user messages are at even indices in history (0, 2, 4...) 
    # due to the pattern [(user, assistant), (user, assistant), ...].
    # selected_index should map correctly to the pairs. If selected_index corresponds to a pair:
    # e.g. selected_index=0 means history[0][0], the user part of the first tuple.
    # This logic may vary depending on how you choose to implement selection.
    # Let's assume selected_index is the index of the tuple in history.
    
    old_user_message, _ = history[selected_index]
    history[selected_index] = (new_text, None)

    # Re-run predict from scratch with the updated history
    # to get a revised assistant response
    response = ""
    for r in predict(history[:selected_index+1], model):
        response = r
    history[selected_index] = (history[selected_index][0], response)
    return history

def update_edit_inputs(history):
    # Populate a dropdown or something similar with user messages to edit
    user_messages = [f"Message {i}: {h[0]}" for i, h in enumerate(history)]
    return gr.update(choices=user_messages)


clients = {}
def chat_with(model):
    with ow.deploy(model) as client:
        clients[model] = client
        with gr.Blocks() as demo:
            history = gr.State([])
            with gr.Row():
                chatbot = gr.Chatbot()
            
            with gr.Row():
                user_input = gr.Textbox(placeholder="Type your message here...")
                submit_btn = gr.Button("Send")

            # UI for editing messages
            with gr.Column():
                message_selector = gr.Dropdown(label="Select a message to edit", choices=[])
                edit_box = gr.Textbox(placeholder="Edit selected message here...")
                edit_button = gr.Button("Apply Edit")

            submit_btn.click(
                add_message,
                inputs=[user_input, history, gr.State(model)],
                outputs=[history, user_input],
                queue=False
            ).then(
                lambda h: h,  # no-op
                inputs=history,
                outputs=chatbot
            )

            # Update message_selector with user messages for editing
            # whenever history changes
            history.change(
                update_edit_inputs,
                inputs=history,
                outputs=message_selector
            )

            edit_button.click(
                edit_message,
                inputs=[message_selector, edit_box, history, gr.State(model)],
                outputs=history
            ).then(
                lambda h: h,
                inputs=history,
                outputs=chatbot
            )

        demo.launch()

if __name__ == '__main__':
    import fire
    fire.Fire(chat_with)
