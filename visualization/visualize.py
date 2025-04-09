import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def load_data(boundary_type):
    """Загрузка данных симуляции для указанного типа границы"""
    results_file = f"../result_simulation/results_{boundary_type}.csv"
    params_file = f"../result_simulation/params_{boundary_type}.csv"
    traj_file = f"../result_simulation/trajectory_{boundary_type}.csv"
    
    results = pd.read_csv(results_file)
    
    # Загружаем параметры
    params_df = pd.read_csv(params_file)
    params = dict(zip(params_df['parameter'], params_df['value']))
    
    # Проверяем наличие файла траектории
    trajectory = None
    if os.path.exists(traj_file):
        trajectory = pd.read_csv(traj_file)
    
    return results, params, trajectory

def plot_thermodynamics():
    """Построение графиков энергии, давления и температуры для всех типов границ"""
    boundary_types = ['periodic', 'reflective', 'mixed']
    colors = ['red', 'blue', 'green']
    
    # Создаем фигуру с тремя подграфиками
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for i, boundary_type in enumerate(boundary_types):
        results, params, _ = load_data(boundary_type)
        
        # Извлекаем целевую температуру из параметров
        t_target = float(params.get('t_target', 0.5))
        
        # Построение графика энергии
        axs[0].plot(results['time'], results['energy'], color=colors[i], label=boundary_type)
        
        # Построение графика давления
        axs[1].plot(results['time'], results['pressure'], color=colors[i], label=boundary_type)
        
        # Построение графика температуры
        axs[2].plot(results['time'], results['temperature'], color=colors[i], label=boundary_type)
    
    # Добавление целевой температуры на график
    axs[2].axhline(y=t_target, color='black', linestyle='--', alpha=0.5, label='Target temp')
    
    # Настройка графиков
    axs[0].set_title('Total Energy')
    axs[0].set_ylabel('Energy')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_title('Pressure')
    axs[1].set_ylabel('Pressure')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].set_title('Temperature')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Temperature')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('thermodynamics_comparison.png', dpi=300)

def create_animation(boundary_type):
    """Создает анимацию движения частиц для указанного типа границы"""
    _, params, trajectory = load_data(boundary_type)
    
    if trajectory is None:
        print(f"Trajectory data for {boundary_type} not found!")
        return
    
    # Определяем количество частиц и кадров
    n_frames = len(trajectory)
    n_particles = (len(trajectory.columns) - 1) // 2
    
    # Извлекаем x и y размеры бокса
    x_size = float(params.get('x', 3.0))
    y_size = float(params.get('y', 3.0))
    
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_title(f'Particle Motion - {boundary_type.capitalize()} Boundaries')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Инициализация пустых данных для частиц
    particles = ax.scatter([], [], s=10, alpha=0.7)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        particles.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return particles, time_text
    
    def update(frame):
        # Извлечение координат всех частиц для текущего кадра
        x_columns = [f'x{i}' for i in range(n_particles)]
        y_columns = [f'y{i}' for i in range(n_particles)]
        
        x_data = trajectory.loc[frame, x_columns].values
        y_data = trajectory.loc[frame, y_columns].values
        
        # Обновление позиций частиц
        particles.set_offsets(np.column_stack([x_data, y_data]))
        
        # Обновление текста времени
        time = frame * float(params.get('dt', 0.0001)) * float(params.get('sampling_interval', 100))
        time_text.set_text(f'Time: {time:.4f}')
        
        return particles, time_text
    
    # Создание анимации
    ani = FuncAnimation(fig, update, frames=range(n_frames),
                        init_func=init, blit=True, interval=50)
    
    # Сохранение анимации
    ani.save(f'animation_{boundary_type}.mp4', writer='ffmpeg', fps=30, dpi=150)
    plt.close()
    print(f"Animation saved as animation_{boundary_type}.mp4")

def main():
    """Основная функция для визуализации всех данных"""
    print("Generating thermodynamics plots...")
    plot_thermodynamics()
    
    print("\nGenerating animations (requires ffmpeg)...")
    for boundary_type in ['periodic', 'reflective', 'mixed']:
        print(f"Creating animation for {boundary_type} boundaries...")
        create_animation(boundary_type)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()