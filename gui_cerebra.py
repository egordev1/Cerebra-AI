#!/usr/bin/env python3
"""
Cerebra AI - Графический интерфейс
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import os
import sys

# Добавляем текущую папку в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cerebra import ai
from cerebra.utils import print_system_info


class CerebraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Cerebra AI")
        self.root.geometry("800x600")
        
        # Загрузка модели
        self.model_loaded = False
        self.load_model()
        
        self.setup_ui()
        
    def load_model(self):
        """Загрузка модели"""
        try:
            ai.load_model("Synthesis-L1")
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Создаем основные фреймы
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить историю", command=self.save_chat_history)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню "Модель"
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Модель", menu=model_menu)
        model_menu.add_command(label="Обучить модель", command=self.open_training_window)
        model_menu.add_command(label="Сохранить модель", command=self.save_model)
        model_menu.add_command(label="Загрузить модель", command=self.load_model)
        
        # Меню "Информация"
        info_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Информация", menu=info_menu)
        info_menu.add_command(label="О системе", command=self.show_system_info)
        
        # Основная область чата
        chat_frame = ttk.Frame(self.root)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # История чата
        self.chat_history = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            height=20
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Ввод пользователя
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)
        
        self.user_input = tk.Text(
            input_frame, 
            height=3, 
            wrap=tk.WORD
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.on_enter_pressed)
        
        send_button = ttk.Button(
            input_frame, 
            text="Отправить", 
            command=self.send_message
        )
        send_button.pack(side=tk.RIGHT)
        
        # Статусная строка
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Готово" if self.model_loaded else "Ошибка загрузки модели"
        )
        self.status_label.pack(anchor=tk.W)
        
        # Приветственное сообщение
        self.add_to_chat("🤖 Cerebra", "Добро пожаловать в Cerebra AI! Я готов к работе.")
    
    def on_enter_pressed(self, event):
        """Обработка нажатия Enter в поле ввода"""
        if event.state & 0x4:  # Ctrl+Enter
            self.user_input.insert(tk.END, "\n")
            return "break"
        else:
            self.send_message()
            return "break"
    
    def send_message(self):
        """Отправка сообщения от пользователя"""
        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text:
            return
            
        self.add_to_chat("👤 Вы", user_text)
        self.user_input.delete("1.0", tk.END)
        
        # Отображаем "печатает..." статус
        self.status_label.config(text="Cerebra печатает...")
        
        # Запускаем обработку сообщения в отдельном потоке
        threading.Thread(target=self.process_message, args=(user_text,), daemon=True).start()
    
    def process_message(self, user_text):
        """Обработка сообщения в отдельном потоке"""
        try:
            response = ai.chat(user_text)
            self.root.after(0, self.add_to_chat, "🤖 Cerebra", response)
            self.root.after(0, self.update_status, "Готово")
        except Exception as e:
            self.root.after(0, self.show_error, f"Ошибка при обработке сообщения: {e}")
            self.root.after(0, self.update_status, "Ошибка")
    
    def add_to_chat(self, sender, message):
        """Добавление сообщения в историю чата"""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"\n{sender}: {message}\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)
    
    def update_status(self, text):
        """Обновление статусной строки"""
        self.status_label.config(text=text)
    
    def show_error(self, message):
        """Показ ошибки"""
        messagebox.showerror("Ошибка", message)
        self.update_status("Ошибка")
    
    def save_chat_history(self):
        """Сохранение истории чата"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    content = self.chat_history.get("1.0", tk.END)
                    f.write(content)
                messagebox.showinfo("Сохранено", "История чата сохранена")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")
    
    def open_training_window(self):
        """Открытие окна обучения модели"""
        training_window = tk.Toplevel(self.root)
        training_window.title("Обучение модели")
        training_window.geometry("400x300")
        
        ttk.Label(training_window, text="Настройки обучения:").pack(pady=10)
        
        # Количество эпох
        epoch_frame = ttk.Frame(training_window)
        epoch_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(epoch_frame, text="Количество эпох:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="5")
        ttk.Entry(epoch_frame, textvariable=self.epochs_var).pack(fill=tk.X, pady=(5, 0))
        
        # Кнопка обучения
        train_button = ttk.Button(
            training_window, 
            text="Начать обучение", 
            command=lambda: self.start_training(training_window)
        )
        train_button.pack(pady=20)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(
            training_window, 
            mode='indeterminate'
        )
        self.progress.pack(fill=tk.X, padx=20, pady=(0, 20))
    
    def start_training(self, window):
        """Запуск обучения модели"""
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError("Количество эпох должно быть положительным")
                
            # Запускаем обучение в отдельном потоке
            self.progress.start()
            threading.Thread(
                target=self.run_training, 
                args=(epochs, window), 
                daemon=True
            ).start()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверное значение: {e}")
    
    def run_training(self, epochs, window):
        """Выполнение обучения в отдельном потоке"""
        try:
            success = ai.real_training(epochs=epochs)
            
            # Обновляем интерфейс в основном потоке
            self.root.after(0, self.progress.stop)
            self.root.after(0, window.destroy)
            
            if success:
                self.root.after(0, messagebox.showinfo, "Обучение", "Обучение завершено успешно!")
                # Обновляем статус
                self.root.after(0, self.update_status, "Модель обучена")
            else:
                self.root.after(0, messagebox.showerror, "Обучение", "Обучение не удалось")
        except Exception as e:
            self.root.after(0, self.progress.stop)
            self.root.after(0, window.destroy)
            self.root.after(0, messagebox.showerror, "Ошибка", f"Ошибка при обучении: {e}")
    
    def save_model(self):
        """Сохранение модели"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch модели", "*.pth"), ("Все файлы", "*.*")]
        )
        if filename:
            try:
                ai.save_model(filename)
                messagebox.showinfo("Сохранено", "Модель успешно сохранена")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {e}")
    
    def show_system_info(self):
        """Показ информации о системе"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Информация о системе")
        info_window.geometry("500x400")
        
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Получаем информацию о системе
        info = ai.info()
        text_widget.insert(tk.END, info)
        text_widget.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = CerebraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()