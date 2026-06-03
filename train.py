import torch
import torch.nn as nn
import torch.optim as optim

model = TinyGPT()

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

print("Training started...")

for step in range(300):  # safe for phone
    x = torch.randint(0, 5000, (1, 32))
    y = torch.randint(0, 5000, (1, 32))

    logits = model(x)

    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 30 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

print("Done")
