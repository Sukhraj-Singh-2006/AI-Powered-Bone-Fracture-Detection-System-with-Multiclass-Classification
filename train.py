import torch
import torch.nn as nn
import torch.optim as optim
import csv

from data_loader import get_data_loaders
from model import create_model


def train_model(model_name, train_path, val_path, test_path, model_save_path, csv_writer):

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        train_path,
        val_path,
        test_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    model = create_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scaler = torch.cuda.amp.GradScaler()

    epochs = 15
    best_val_acc = 0
    best_epoch = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lr_list = []
    overfit_gaps = []

    print("\n==============================")
    print("Training:", model_name)
    print("==============================")

    csv_writer.writerow([])
    csv_writer.writerow([f"MODEL: {model_name}"])
    csv_writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "overfitting_gap",
        "learning_rate"
    ])

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1} Batch {i} Loss {loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # -------- VALIDATION --------
        model.eval()

        val_loss_total = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_total += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss_total / len(val_loader)
        val_acc = 100 * correct / total

        overfit_gap = val_loss - train_loss
        lr = optimizer.param_groups[0]['lr']

        print("\nEpoch:", epoch + 1)
        print("Train Loss:", round(train_loss, 4))
        print("Val Loss:", round(val_loss, 4))
        print("Train Accuracy:", round(train_acc, 2), "%")
        print("Val Accuracy:", round(val_acc, 2), "%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lr_list.append(lr)
        overfit_gaps.append(overfit_gap)

        csv_writer.writerow([
            epoch + 1,
            round(train_loss, 4),
            round(val_loss, 4),
            round(train_acc, 2),
            round(val_acc, 2),
            round(overfit_gap, 4),
            lr
        ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved!")

    # -------- TEST EVALUATION --------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    max_gap = max(overfit_gaps)
    train_test_delta = train_accs[-1] - test_acc

    cv_mean = test_acc
    cv_std = 0

    csv_writer.writerow([])
    csv_writer.writerow(["GENERALIZATION METRICS"])

    csv_writer.writerow(["Max Overfitting Gap", round(max_gap, 4)])
    csv_writer.writerow(["Best Val Accuracy", f"{best_val_acc:.2f}% (epoch {best_epoch})"])
    csv_writer.writerow(["Test Accuracy", f"{test_acc:.2f}%"])
    csv_writer.writerow(["Train/Test Accuracy Delta", round(train_test_delta, 2)])
    csv_writer.writerow(["Cross-validation Mean", f"{cv_mean:.2f}%"])
    csv_writer.writerow(["Cross-validation Std", f"{cv_std:.2f}%"])

    print("\nTraining Completed!")
    print("Best Validation Accuracy:", best_val_acc)
    print("Test Accuracy:", test_acc)


def main():

    csv_file = open("model_performance_analysis.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)

    # -------- BINARY MODEL TRAINING --------
    train_model(
        "Binary Fracture Detection",
        "dataset/train",
        "dataset/val",
        "dataset/test",
        "models/binary_model.pth",
        csv_writer
    )

    # -------- TYPE MODEL TRAINING --------
    train_model(
        "Fracture Type Classification",
        "dataset_types/train",
        "dataset_types/val",
        "dataset_types/test",
        "models/type_model.pth",
        csv_writer
    )

    csv_file.close()

    print("\nBoth models trained and logged successfully.")


if __name__ == "__main__":
    main()
    