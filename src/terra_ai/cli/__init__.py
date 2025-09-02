import typer
from pathlib import Path
from terra_ai.config import settings

app = typer.Typer(help="Terra AI CLI")

@app.command()
def dataset(name: str, to: str | None = None):
    """
    Скачать учебный датасет. Поддержка: digits, imdb (минимум), mnist (если vision-экстра).
    """
    target = Path(to) if to else settings.data_dir / name
    target.mkdir(parents=True, exist_ok=True)
    if name.lower() == "digits":
        from sklearn.datasets import load_digits
        import numpy as np
        ds = load_digits()
        np.savez_compressed(target / "digits.npz", data=ds.data, target=ds.target)
        typer.echo(f"Saved digits to {target}")
    elif name.lower() == "mnist":
        try:
            from torchvision import datasets, transforms
        except Exception:
            raise typer.Exit("MNIST требует extra 'vision': pip install terra-ai[vision]")
        datasets.MNIST(str(target), train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST(str(target), train=False, download=True, transform=transforms.ToTensor())
        typer.echo(f"Saved MNIST to {target}")
    else:
        raise typer.Exit("Неизвестный датасет: digits | mnist")

@app.command()
def train(spec: str = typer.Argument(..., help="Спецификация модели (e.g. 'LogReg' или 'CNN@mnist')"),
          dataset: str = "digits",
          epochs: int = 5):
    """
    Мини-тренировка для курса: digits -> LogReg (CPU), mnist -> простая CNN (если доступен torch).
    """
    if dataset.lower() == "digits":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        import numpy as np
        data = np.load(settings.data_dir / "digits" / "digits.npz")
        X, y = data["data"], data["target"]
        split = int(0.8 * len(X))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[:split], y[:split])
        acc = accuracy_score(y[split:], clf.predict(X[split:]))
        typer.echo(f"[digits] LogReg accuracy: {acc:.3f}")
    elif dataset.lower() == "mnist":
        try:
            import torch, torch.nn as nn, torch.optim as optim
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
        except Exception:
            raise typer.Exit("MNIST-CNN требует extra 'vision': pip install terra-ai[vision]")
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
                )
            def forward(self, x): return self.net(x)
        train_ds = datasets.MNIST(str(settings.data_dir / "mnist"), train=True, download=True, transform=transforms.ToTensor())
        test_ds  = datasets.MNIST(str(settings.data_dir / "mnist"), train=False, download=True, transform=transforms.ToTensor())
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_dl  = DataLoader(test_ds, batch_size=256)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinyCNN().to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss = nn.CrossEntropyLoss()
        for e in range(epochs):
            model.train()
            for x,y in train_dl:
                x,y = x.to(device), y.to(device)
                opt.zero_grad(); l = loss(model(x), y); l.backward(); opt.step()
            typer.echo(f"epoch {e+1}/{epochs} done")
        model.eval()
        correct,total = 0,0
        import torch
        with torch.no_grad():
            for x,y in test_dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        typer.echo(f"[mnist] TinyCNN accuracy: {correct/total:.3f}")
    else:
        raise typer.Exit("Поддержка train: digits | mnist")
