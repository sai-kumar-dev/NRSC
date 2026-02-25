from core import logger
from memory_utils import clear_memory
from physics import physics_informed_loss
import torch.optim as optim
import torch

# Training function
def train_pinn(model, train_loader, val_loader, num_epochs=70, learning_rate=0.001,
               lambda_physics=0.1, device='cpu'):
    """Train the PINN model"""

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                    patience=10, verbose=True)

    train_losses = []
    val_losses = []

    logger.info(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        train_data_loss_epoch = 0.0
        train_physics_loss_epoch = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            total_loss, data_loss, physics_loss = physics_informed_loss(
                model, X_batch, y_batch, lambda_physics)

            total_loss.backward()
            optimizer.step()

            train_loss_epoch += total_loss.item()
            train_data_loss_epoch += data_loss.item()
            train_physics_loss_epoch += physics_loss.item()

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                total_loss, _, _ = physics_informed_loss(model, X_batch, y_batch, lambda_physics)
                val_loss_epoch += total_loss.item()

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_data_loss = train_data_loss_epoch / len(train_loader)
        avg_physics_loss = train_physics_loss_epoch / len(train_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.6f} (Data: {avg_data_loss:.6f}, Physics: {avg_physics_loss:.6f})")
            logger.info(f"  Val Loss: {avg_val_loss:.6f}")

        clear_memory()

    return train_losses, val_losses
