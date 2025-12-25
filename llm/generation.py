
def generate(tok, model, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return tok.decode(out[0], skip_special_tokens=True)
