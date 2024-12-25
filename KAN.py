import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# تحميل البيانات والتحويلات
class BrainTumorDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []  # قائمة لتخزين الصور
        self.labels = []  # قائمة لتخزين التصنيفات
        self.classes = os.listdir(directory)  # استخراج الفئات من المسار
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # تعيين فئة لكل نوع
        
        # تحميل الصور من كل فئة
        for cls in self.classes:
            cls_folder = os.path.join(directory, cls)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')  # تحويل الصورة إلى RGB
                    if self.transform:
                        image = self.transform(image)  # تطبيق التحويلات إذا كانت موجودة
                    self.images.append(image)
                    self.labels.append(self.class_to_idx[cls])  # إضافة التصنيف
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# تعريف التحويلات على الصور (تغيير الحجم، تحويل إلى Tensor، والتطبيع)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # تغيير حجم الصور إلى 128x128
    transforms.ToTensor(),  # تحويل الصورة إلى Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # تطبيع الصور للقيم بين -1 و 1
])

# مسار بيانات التدريب والاختبار
train_dir = "/home/mohamed/AI Engineering/Medical Project/Data set/Training"
test_dir = "/home/mohamed/AI Engineering/Medical Project/Data set/Testing"

# تحميل البيانات
train_dataset = BrainTumorDataset(train_dir, transform)
test_dataset = BrainTumorDataset(test_dir, transform)

# تعريف طبقة Kolmogorov-Arnold Network (KAN) مع توسعات متعددة الحدود
class KANLayer(nn.Module):
    def __init__(self, input_dim):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim  # عدد الأبعاد المدخلة

    def forward(self, x):
        # تطبيق التوسع متعدد الحدود ودوال غير خطية (مثل sin, cos, tanh)
        x1 = torch.pow(x, 2)  # توسيع باستخدام x^2
        x2 = torch.pow(x, 3)  # توسيع باستخدام x^3
        sin_x = torch.sin(x)  # تطبيق دالة sin
        cos_x = torch.cos(x)  # تطبيق دالة cos
        tanh_x = torch.tanh(x)  # تطبيق دالة tanh
        
        # دمج جميع المدخلات بعد التوسع
        expanded_x = torch.cat([x1, x2, sin_x, cos_x, tanh_x], dim=1)
        return expanded_x

# تعريف طبقة B-spline
class BSplineLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BSplineLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # تعريف النقاط الخاصة بـ B-splines (مثال: 4 نقاط تحكم)
        self.control_points = nn.Parameter(torch.randn(4, input_dim))  # عدد النقاط يمكن تغييره

    def forward(self, x):
        # تطبيق B-splines باستخدام النقاط للتحكم
        spline_values = self.b_spline_interpolation(x)
        return spline_values

    def b_spline_interpolation(self, x):
        # هنا نقوم بحساب القيم باستخدام B-splines
        spline_result = torch.matmul(x, self.control_points.T)
        return spline_result

# تعريف نموذج KAN مع طبقة B-spline
class KANModelWithBSplines(nn.Module):
    def __init__(self, num_classes):
        super(KANModelWithBSplines, self).__init__()
        self.kan_layer = KANLayer(input_dim=3 * 128 * 128)  # طبقة KAN
        # استخدام BSplineLayer بدلاً من nn.Linear
        self.b_spline_layer = BSplineLayer(input_dim=5 * 3 * 128 * 128, num_classes=num_classes)  # طبقة B-spline

    def forward(self, x):
        x = x.view(-1, 3 * 128 * 128)  # تسطيح الصورة
        x = self.kan_layer(x)  # تطبيق طبقة KAN
        x = self.b_spline_layer(x)  # تطبيق طبقة B-spline
        return x

# تعريف المحسن ودالة الخسارة
model = KANModelWithBSplines(len(train_dataset.class_to_idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# تحميل البيانات
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# تعريف القوائم لتخزين الخسارة والدقة
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# تدريب النموذج
for epoch in range(50):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # التدريب
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # حساب دقة التدريب
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch+1}/50 - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

    # تقليل معدل التعلم إذا كانت الخسارة لا تتحسن
    scheduler.step(avg_train_loss)

    # التحقق (Validation)
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_val_loss += loss.item()

            # حساب دقة التحقق
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = running_val_loss / len(test_loader)
    val_accuracy = correct_val / total_val

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/50 - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# رسم خسارة ودقة التدريب والتحقق
plt.figure(figsize=(12, 5))

# رسم الخسارة
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'g-o', label='Training Loss')
plt.plot(val_losses, 'r-o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# رسم الدقة
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-o', label='Training Accuracy')
plt.plot(val_accuracies, 'r-o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# تقييم النموذج
model.eval()
true_classes = []
predicted_classes = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_classes.extend(labels.cpu().numpy())
        predicted_classes.extend(predicted.cpu().numpy())

# مصفوفة التشويش
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# تقرير التصنيف
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=train_dataset.classes))

# دقة الاختبار
test_accuracy = np.sum(np.array(true_classes) == np.array(predicted_classes)) / len(true_classes)
print(f"Test Accuracy: {test_accuracy:.4f}")