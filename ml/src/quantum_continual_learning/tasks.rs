//! Task management for quantum continual learning

use std::collections::HashMap;
use super::config::ContinualTask;

/// Task sequence management
#[derive(Debug)]
pub struct TaskSequence {
    tasks: HashMap<usize, ContinualTask>,
    task_order: Vec<usize>,
}

impl TaskSequence {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            task_order: Vec::new(),
        }
    }
    
    pub fn add_task(&mut self, task: ContinualTask) {
        let task_id = task.task_id;
        self.tasks.insert(task_id, task);
        if !self.task_order.contains(&task_id) {
            self.task_order.push(task_id);
        }
    }
    
    pub fn get_task(&self, task_id: usize) -> Option<&ContinualTask> {
        self.tasks.get(&task_id)
    }
    
    pub fn get_all_tasks(&self) -> &HashMap<usize, ContinualTask> {
        &self.tasks
    }
    
    pub fn get_task_order(&self) -> &Vec<usize> {
        &self.task_order
    }
}