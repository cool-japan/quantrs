//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{
    BuilderConfig, ComparisonOperator, ConstraintType, ExportFormat, LabelPosition, LabelSettings,
    ObjectiveExpression, OptimizationDirection, Position, ProblemValidator, ValidationSeverity,
    VariableDomain, VariableShape, VariableType, VariableVisualProperties, VisualProblem,
    VisualProblemBuilder, VisualVariable,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_visual_problem_builder() -> Result<(), String> {
        let config = BuilderConfig::default();
        let mut builder = VisualProblemBuilder::new(config);
        builder.new_problem("Test Problem")?;
        let var1_id = builder.add_variable(
            "x1",
            VariableType::Binary,
            Position {
                x: 100.0,
                y: 100.0,
                z: None,
            },
        )?;
        let var2_id = builder.add_variable(
            "x2",
            VariableType::Binary,
            Position {
                x: 200.0,
                y: 100.0,
                z: None,
            },
        )?;
        assert_eq!(builder.problem().variables.len(), 2);
        let _constraint_id = builder.add_constraint(
            "Sum constraint",
            ConstraintType::Linear {
                coefficients: vec![1.0, 1.0],
                operator: ComparisonOperator::LessEqual,
                rhs: 1.0,
            },
            vec![var1_id.clone(), var2_id.clone()],
        )?;
        assert_eq!(builder.problem().constraints.len(), 1);
        let mut coefficients = HashMap::new();
        coefficients.insert(var1_id, 1.0);
        coefficients.insert(var2_id, 2.0);
        builder.set_objective(
            "Linear objective",
            ObjectiveExpression::Linear {
                coefficients,
                constant: 0.0,
            },
            OptimizationDirection::Maximize,
        )?;
        assert!(builder.problem().objective.is_some());
        builder.undo()?;
        assert!(builder.problem().objective.is_none());
        builder.redo()?;
        assert!(builder.problem().objective.is_some());
        let python_code = builder.generate_code(ExportFormat::Python)?;
        assert!(python_code.contains("symbols"));
        assert!(python_code.contains("SASampler"));
        let json = builder.save_problem()?;
        assert!(json.contains("Test Problem"));
        Ok(())
    }
    #[test]
    fn test_validation() -> Result<(), String> {
        let mut validator = ProblemValidator::new();
        let mut problem = VisualProblem::new();
        let errors = validator.validate(&problem)?;
        assert!(!errors.is_empty());
        problem.variables.push(VisualVariable {
            id: "var1".to_string(),
            name: "x1".to_string(),
            var_type: VariableType::Binary,
            domain: VariableDomain::Binary,
            position: Position {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            visual_properties: VariableVisualProperties {
                color: "#000000".to_string(),
                size: 10.0,
                shape: VariableShape::Circle,
                visible: true,
                label: LabelSettings {
                    show: true,
                    text: None,
                    font_size: 12.0,
                    position: LabelPosition::Bottom,
                },
            },
            description: String::new(),
            groups: Vec::new(),
        });
        let errors = validator.validate(&problem)?;
        assert!(errors
            .iter()
            .any(|e| matches!(e.severity, ValidationSeverity::Warning)));
        Ok(())
    }
}
