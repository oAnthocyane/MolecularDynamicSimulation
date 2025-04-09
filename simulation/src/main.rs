use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::time::Instant;
use std::fs::File;
use std::io::{Write, BufWriter};

// Структура для хранения параметров симуляции
#[derive(Clone)]
struct SimulationParams {
    x: f64,
    y: f64,
    m: f64,
    sigma: f64,
    epsilon: f64,
    n: usize,
    k_b: f64,
    t_target: f64,
    dt: f64,
    num_steps: usize,
    sampling_interval: usize,
    thermostat_tau: f64,
    rcut: f64,
    rmin: f64,
    use_thermostat: bool,
    use_cell_lists: bool,
    boundary_type: String,
    save_trajectory: bool,
}

impl Default for SimulationParams {
    fn default() -> Self {
        let sigma = 0.2;
        SimulationParams {
            x: 5.0,
            y: 20.0,
            m: 0.5,
            sigma,
            epsilon: 0.8,
            n: 600,
            k_b: 1.5,
            t_target: 0.5,
            dt: 0.0001,
            num_steps: 10000,
            sampling_interval: 100,
            thermostat_tau: 0.1,
            rcut: 2.0 * sigma,
            rmin: 0.9 * sigma,
            use_thermostat: false,
            use_cell_lists: true,
            boundary_type: String::from("periodic"),
            save_trajectory: false,
        }
    }
}

// Структура для результатов симуляции
struct SimulationResults {
    times: Vec<f64>,
    energies: Vec<f64>,
    pressures: Vec<f64>,
    temperatures: Vec<f64>,
    trajectory: Option<Vec<Array2<f64>>>,
}

fn initialize_positions(n: usize, x: f64, y: f64) -> Array2<f64> {
    let n_side_x = (((n as f64) * x / y).sqrt().ceil()) as usize;
    let n_side_y = (((n as f64) * y / x).sqrt().ceil()) as usize;
    let spacing_x = x / (n_side_x as f64);
    let spacing_y = y / (n_side_y as f64);

    let mut positions = Array2::<f64>::zeros((n, 2));
    let mut index = 0;

    for i in 0..n_side_x {
        for j in 0..n_side_y {
            if index < n {
                positions[[index, 0]] = (i as f64) * spacing_x + spacing_x / 2.0;
                positions[[index, 1]] = (j as f64) * spacing_y + spacing_y / 2.0;
                index += 1;
            }
        }
    }

    positions
}

fn initialize_velocities(n: usize, k_b: f64, t_target: f64, m: f64) -> Array2<f64> {
    let std_dev = (k_b * t_target / m).sqrt();
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, std_dev).unwrap();
    
    let mut velocities = Array2::<f64>::random_using((n, 2), normal, &mut rng);
    
    // Удаляем суммарный импульс
    let mean_x = velocities.column(0).mean().unwrap();
    let mean_y = velocities.column(1).mean().unwrap();
    
    for i in 0..n {
        velocities[[i, 0]] -= mean_x;
        velocities[[i, 1]] -= mean_y;
    }
    
    velocities
}

fn build_cell_lists(
    positions: &Array2<f64>,
    n: usize,
    cell_size: f64,
    n_cells_x: usize,
    n_cells_y: usize,
) -> Vec<Vec<usize>> {
    let mut cell_lists = vec![Vec::new(); n_cells_x * n_cells_y];
    
    for i in 0..n {
        let cell_x = (positions[[i, 0]] / cell_size).floor() as usize;
        let cell_y = (positions[[i, 1]] / cell_size).floor() as usize;
        
        if cell_x < n_cells_x && cell_y < n_cells_y {
            cell_lists[cell_x + cell_y * n_cells_x].push(i);
        }
    }
    
    cell_lists
}

fn get_neighbor_list(
    particle_index: usize,
    positions: &Array2<f64>,
    cell_size: f64,
    n_cells_x: usize,
    n_cells_y: usize,
    cell_lists: &[Vec<usize>],
) -> Vec<usize> {
    let cell_x = (positions[[particle_index, 0]] / cell_size).floor() as usize;
    let cell_y = (positions[[particle_index, 1]] / cell_size).floor() as usize;
    let mut neighbors = Vec::new();
    
    for dx in [n_cells_x - 1, 0, 1].iter() {
        for dy in [n_cells_y - 1, 0, 1].iter() {
            let nx = (cell_x + dx) % n_cells_x;
            let ny = (cell_y + dy) % n_cells_y;
            
            if nx < n_cells_x && ny < n_cells_y {
                neighbors.extend(cell_lists[nx + ny * n_cells_x].iter());
            }
        }
    }
    
    neighbors
}

fn compute_forces(
    positions: &Array2<f64>,
    n: usize,
    x: f64,
    y: f64,
    sigma: f64,
    epsilon: f64,
    rcut: f64,
    rmin: f64,
    use_cell_lists: bool,
    cell_size: f64,
    n_cells_x: usize,
    n_cells_y: usize,
) -> (Array2<f64>, f64) {
    let mut forces = Array2::<f64>::zeros((n, 2));
    let mut potential_energy = 0.0;
    let shift = 4.0 * epsilon * ((sigma / rcut).powi(12) - (sigma / rcut).powi(6));
    
    let cell_lists = if use_cell_lists {
        Some(build_cell_lists(positions, n, cell_size, n_cells_x, n_cells_y))
    } else {
        None
    };
    
    for i in 0..n {
        let neighbors = if let Some(ref lists) = cell_lists {
            get_neighbor_list(i, positions, cell_size, n_cells_x, n_cells_y, lists)
        } else {
            (i+1..n).collect()
        };
        
        for &j in &neighbors {
            if i == j {
                continue;
            }
            
            let mut r_vec = Array1::<f64>::zeros(2);
            r_vec[0] = positions[[j, 0]] - positions[[i, 0]];
            r_vec[1] = positions[[j, 1]] - positions[[i, 1]];
            
            // Периодические граничные условия
            r_vec[0] -= (r_vec[0] / x).round() * x;
            r_vec[1] -= (r_vec[1] / y).round() * y;
            
            let r = (r_vec[0].powi(2) + r_vec[1].powi(2)).sqrt();
            
            let f_mag: f64;
            if r < rmin {
                // Отталкивающая сила на минимальном расстоянии
                f_mag = 24.0 * epsilon * (2.0 * (sigma / rmin).powi(12) - (sigma / rmin).powi(6)) / rmin;
                potential_energy += 4.0 * epsilon * ((sigma / rmin).powi(12) - (sigma / rmin).powi(6)) - shift;
            } else if r < rcut {
                // Стандартный случай
                let sr6 = (sigma / r).powi(6);
                f_mag = 24.0 * epsilon * (2.0 * sr6.powi(2) - sr6) / r;
                potential_energy += 4.0 * epsilon * (sr6.powi(2) - sr6) - shift;
            } else {
                continue;
            }
            
            forces[[i, 0]] += f_mag * r_vec[0] / r;
            forces[[i, 1]] += f_mag * r_vec[1] / r;
            forces[[j, 0]] -= f_mag * r_vec[0] / r;
            forces[[j, 1]] -= f_mag * r_vec[1] / r;
        }
    }
    
    (forces, potential_energy)
}

fn apply_thermostat(
    velocities: &mut Array2<f64>,
    n: usize,
    k_b: f64,
    t_target: f64,
    dt: f64,
    thermostat_tau: f64,
    m: f64,
) {
    let mut current_temp = 0.0;
    for i in 0..n {
        current_temp += m * (velocities[[i, 0]].powi(2) + velocities[[i, 1]].powi(2));
    }
    current_temp /= 2.0 * (n as f64) * k_b;
    
    let scale = (1.0 + (dt / thermostat_tau) * (t_target / current_temp - 1.0)).sqrt();
    
    for i in 0..n {
        velocities[[i, 0]] *= scale;
        velocities[[i, 1]] *= scale;
    }
}

fn apply_boundary_conditions(
    positions: &mut Array2<f64>,
    velocities: &mut Array2<f64>,
    x: f64,
    y: f64,
    boundary_type: &str,
    n: usize,
) {
    match boundary_type {
        "periodic" => {
            for i in 0..n {
                positions[[i, 0]] = positions[[i, 0]] % x;
                positions[[i, 1]] = positions[[i, 1]] % y;
                
                // Обработка отрицательных значений
                if positions[[i, 0]] < 0.0 {
                    positions[[i, 0]] += x;
                }
                if positions[[i, 1]] < 0.0 {
                    positions[[i, 1]] += y;
                }
            }
        },
        "reflective" => {
            for i in 0..n {
                // X dimension
                if positions[[i, 0]] < 0.0 {
                    positions[[i, 0]] *= -1.0;
                    velocities[[i, 0]] *= -1.0;
                } else if positions[[i, 0]] > x {
                    positions[[i, 0]] = 2.0 * x - positions[[i, 0]];
                    velocities[[i, 0]] *= -1.0;
                }
                
                // Y dimension
                if positions[[i, 1]] < 0.0 {
                    positions[[i, 1]] *= -1.0;
                    velocities[[i, 1]] *= -1.0;
                } else if positions[[i, 1]] > y {
                    positions[[i, 1]] = 2.0 * y - positions[[i, 1]];
                    velocities[[i, 1]] *= -1.0;
                }
            }
        },
        "mixed" => {
            for i in 0..n {
                // X dimension - reflective
                if positions[[i, 0]] < 0.0 {
                    positions[[i, 0]] = 0.0;
                    velocities[[i, 0]] *= -1.0;
                } else if positions[[i, 0]] > x {
                    positions[[i, 0]] = x;
                    velocities[[i, 0]] *= -1.0;
                }
                
                // Y dimension - periodic
                positions[[i, 1]] = positions[[i, 1]] % y;
                if positions[[i, 1]] < 0.0 {
                    positions[[i, 1]] += y;
                }
            }
        },
        _ => panic!("Unknown boundary type: {}", boundary_type),
    }
}

// Функция для сохранения результатов в CSV файл
use std::fs::{self};
use std::path::Path;

fn save_results_to_csv(
    results: &SimulationResults,
    boundary_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Создаем директорию, если не существует
    let dir_path = Path::new("../result_simulation");
    if !dir_path.exists() {
        fs::create_dir_all(dir_path)?;
    }

    // Сохраняем основные данные
    let filename = dir_path.join(format!("results_{}.csv", boundary_type));
    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "time,energy,pressure,temperature")?;
    for i in 0..results.times.len() {
        writeln!(
            writer, 
            "{},{},{},{}",
            results.times[i],
            results.energies[i],
            results.pressures[i],
            results.temperatures[i]
        )?;
    }
    println!("Saved data to: {:?}", filename);

    // Сохраняем траекторию
    if let Some(ref trajectory) = results.trajectory {
        let traj_filename = dir_path.join(format!("trajectory_{}.csv", boundary_type));
        let traj_file = File::create(&traj_filename)?;
        let mut traj_writer = BufWriter::new(traj_file);
        
        // Заголовок с номерами частиц
        write!(traj_writer, "frame")?;
        if let Some(first_frame) = trajectory.get(0) {
            for i in 0..first_frame.shape()[0] {
                write!(traj_writer, ",x{},y{}", i, i)?;
            }
        }
        writeln!(traj_writer)?;
        
        // Данные траектории
        for (frame, positions) in trajectory.iter().enumerate() {
            write!(traj_writer, "{}", frame)?;
            for i in 0..positions.shape()[0] {
                write!(traj_writer, ",{},{}", positions[[i, 0]], positions[[i, 1]])?;
            }
            writeln!(traj_writer)?;
        }
        println!("Saved trajectory to: {:?}", traj_filename);
    }

    Ok(())
}

fn save_simulation_params(
    params: &SimulationParams,
    boundary_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("../result_simulation/params_{}.csv", boundary_type);
    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "parameter,value")?;
    writeln!(writer, "boundary_type,{}", params.boundary_type)?;
    writeln!(writer, "x,{}", params.x)?;
    writeln!(writer, "y,{}", params.y)?;
    writeln!(writer, "m,{}", params.m)?;
    writeln!(writer, "sigma,{}", params.sigma)?;
    writeln!(writer, "epsilon,{}", params.epsilon)?;
    writeln!(writer, "n,{}", params.n)?;
    writeln!(writer, "k_b,{}", params.k_b)?;
    writeln!(writer, "t_target,{}", params.t_target)?;
    writeln!(writer, "dt,{}", params.dt)?;
    writeln!(writer, "num_steps,{}", params.num_steps)?;
    writeln!(writer, "sampling_interval,{}", params.sampling_interval)?;
    writeln!(writer, "thermostat_tau,{}", params.thermostat_tau)?;
    writeln!(writer, "rcut,{}", params.rcut)?;
    writeln!(writer, "rmin,{}", params.rmin)?;
    writeln!(writer, "use_thermostat,{}", params.use_thermostat)?;
    writeln!(writer, "use_cell_lists,{}", params.use_cell_lists)?;
    
    println!("Saved parameters to: {}", filename);
    Ok(())
}

fn simulate(params: SimulationParams) -> SimulationResults {
    let SimulationParams {
        x, y, m, sigma, epsilon, n, k_b, t_target, dt, num_steps,
        sampling_interval, thermostat_tau, rcut, rmin, use_thermostat,
        use_cell_lists, boundary_type, save_trajectory
    } = params;
    
    // Cell list parameters
    let cell_size = rcut;
    let n_cells_x = (x / cell_size).floor() as usize;
    let n_cells_y = (y / cell_size).floor() as usize;
    
    // Initialize simulation
    let mut positions = initialize_positions(n, x, y);
    let mut velocities = initialize_velocities(n, k_b, t_target, m);
    let (mut forces, mut potential_energy) = compute_forces(
        &positions, n, x, y, sigma, epsilon, rcut, rmin,
        use_cell_lists, cell_size, n_cells_x, n_cells_y
    );
    
    // Storage for data
    let mut energies = Vec::new();
    let mut pressures = Vec::new();
    let mut temperatures = Vec::new();
    let mut trajectory = if save_trajectory { Some(Vec::new()) } else { None };
    
    let start_time = Instant::now();
    
    println!("{} Simulation", boundary_type);
    
    for step in 0..num_steps {
        // Velocity Verlet integration
        // Update positions
        for i in 0..n {
            positions[[i, 0]] += velocities[[i, 0]] * dt + 0.5 * forces[[i, 0]] / m * dt.powi(2);
            positions[[i, 1]] += velocities[[i, 1]] * dt + 0.5 * forces[[i, 1]] / m * dt.powi(2);
        }
        
        apply_boundary_conditions(&mut positions, &mut velocities, x, y, &boundary_type, n);
        
        let (new_forces, new_potential_energy) = compute_forces(
            &positions, n, x, y, sigma, epsilon, rcut, rmin,
            use_cell_lists, cell_size, n_cells_x, n_cells_y
        );
        
        // Update velocities
        for i in 0..n {
            velocities[[i, 0]] += 0.5 * (forces[[i, 0]] + new_forces[[i, 0]]) / m * dt;
            velocities[[i, 1]] += 0.5 * (forces[[i, 1]] + new_forces[[i, 1]]) / m * dt;
        }
        
        forces = new_forces;
        potential_energy = new_potential_energy;
        
        // Apply thermostat
        if use_thermostat {
            apply_thermostat(&mut velocities, n, k_b, t_target, dt, thermostat_tau, m);
        }
        
        // Sample properties
        if (step + 1) % sampling_interval == 0 {
            let mut kinetic_energy = 0.0;
            for i in 0..n {
                kinetic_energy += 0.5 * m * (velocities[[i, 0]].powi(2) + velocities[[i, 1]].powi(2));
            }
            
            let temperature = kinetic_energy / (n as f64 * k_b);
            
            let mut virial = 0.0;
            for i in 0..n {
                virial += forces[[i, 0]] * positions[[i, 0]] + forces[[i, 1]] * positions[[i, 1]];
            }
            
            let pressure = (2.0 * n as f64 * k_b * temperature + virial) / (x * y);
            
            energies.push(kinetic_energy + potential_energy);
            pressures.push(pressure);
            temperatures.push(temperature);
            
            if let Some(ref mut traj) = trajectory {
                traj.push(positions.clone());
            }
            
            // Вывод прогресса
            if (step + 1) % (num_steps / 10) == 0 {
                println!("Step {}/{} ({}%)", step + 1, num_steps, (step + 1) * 100 / num_steps);
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("Simulation finished in {:.2?}", elapsed);
    
    let times: Vec<f64> = (0..energies.len())
        .map(|i| (i as f64) * dt * (sampling_interval as f64))
        .collect();
    
    SimulationResults {
        times,
        energies,
        pressures,
        temperatures,
        trajectory,
    }
}

fn run_simulations() -> Result<(), Box<dyn std::error::Error>> {
    let mut base_params = SimulationParams::default();
    // Опционально можно включить сохранение траектории
    base_params.save_trajectory = true;
    
    println!("\n=== LJTS with Periodic Boundaries ===");
    let mut params1 = base_params.clone();
    params1.boundary_type = String::from("periodic");
    let results1 = simulate(params1.clone());
    save_results_to_csv(&results1, "periodic")?;
    save_simulation_params(&params1, "periodic")?;
    
    println!("\n=== LJTS with Reflective Boundaries ===");
    let mut params2 = base_params.clone();
    params2.boundary_type = String::from("reflective");
    let results2 = simulate(params2.clone());
    save_results_to_csv(&results2, "reflective")?;
    save_simulation_params(&params2, "reflective")?;
    
    println!("\n=== LJTS with Mixed Boundaries ===");
    let mut params3 = base_params.clone();
    params3.boundary_type = String::from("mixed");
    let results3 = simulate(params3.clone());
    save_results_to_csv(&results3, "mixed")?;
    save_simulation_params(&params3, "mixed")?;
    
    println!("\nAll simulations completed. Results saved to CSV files.");
    println!("Use the accompanying Python script to visualize the results.");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_simulations()?;
    Ok(())
}