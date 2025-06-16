//! Algorithm Registry for Quantum Algorithm Marketplace
//!
//! This module manages the registration, storage, and metadata of quantum algorithms
//! in the marketplace, including version control, categorization, and search indexing.

use super::*;

/// Algorithm registry for managing all registered algorithms
pub struct AlgorithmRegistry {
    config: RegistryConfig,
    algorithms: HashMap<String, RegisteredAlgorithm>,
    categories: HashMap<String, Vec<String>>,
    tags: HashMap<String, HashSet<String>>,
    search_index: SearchIndex,
    algorithm_dependencies: HashMap<String, Vec<String>>,
    algorithm_performance: HashMap<String, AlgorithmPerformanceData>,
}

/// Registered algorithm in the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredAlgorithm {
    pub algorithm_id: String,
    pub metadata: AlgorithmMetadata,
    pub code: AlgorithmCode,
    pub documentation: AlgorithmDocumentation,
    pub test_suite: AlgorithmTestSuite,
    pub performance_benchmarks: Vec<PerformanceBenchmark>,
    pub licensing: LicensingInfo,
    pub registration_info: RegistrationInfo,
    pub usage_statistics: UsageStatistics,
}

/// Algorithm metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub author_email: String,
    pub organization: Option<String>,
    pub category: AlgorithmCategory,
    pub subcategory: Option<String>,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub complexity_class: ComplexityClass,
    pub quantum_advantage: QuantumAdvantage,
    pub hardware_requirements: HardwareRequirements,
    pub dependencies: Vec<Dependency>,
}

/// Algorithm categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmCategory {
    Optimization,
    MachineLearning,
    Cryptography,
    Simulation,
    Chemistry,
    Finance,
    Logistics,
    SearchAndDatabase,
    ErrorCorrection,
    Characterization,
    Benchmarking,
    Utility,
    Educational,
    Research,
    Custom(String),
}

/// Complexity classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityClass {
    pub time_complexity: String,
    pub space_complexity: String,
    pub quantum_complexity: String,
    pub classical_preprocessing: Option<String>,
    pub classical_postprocessing: Option<String>,
}

/// Quantum advantage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantage {
    pub advantage_type: AdvantageType,
    pub speedup_factor: Option<f64>,
    pub problem_size_threshold: Option<usize>,
    pub verification_method: String,
    pub theoretical_basis: String,
    pub experimental_validation: bool,
}

/// Types of quantum advantage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdvantageType {
    Exponential,
    Polynomial,
    Quadratic,
    Constant,
    Unknown,
    Disputed,
}

/// Hardware requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub min_qubits: usize,
    pub recommended_qubits: usize,
    pub max_circuit_depth: usize,
    pub required_gates: Vec<String>,
    pub connectivity_requirements: ConnectivityRequirements,
    pub fidelity_requirements: FidelityRequirements,
    pub supported_platforms: Vec<String>,
    pub special_hardware: Vec<String>,
}

/// Connectivity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityRequirements {
    pub topology_type: TopologyType,
    pub connectivity_degree: Option<usize>,
    pub all_to_all_required: bool,
    pub specific_connections: Vec<(usize, usize)>,
}

/// Topology types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologyType {
    Linear,
    Ring,
    Grid2D,
    Grid3D,
    AllToAll,
    Star,
    Tree,
    Random,
    Custom,
}

/// Fidelity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityRequirements {
    pub min_gate_fidelity: f64,
    pub min_readout_fidelity: f64,
    pub min_state_preparation_fidelity: f64,
    pub coherence_time_requirement: Duration,
    pub error_budget: f64,
}

/// Algorithm dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub dependency_type: DependencyType,
    pub optional: bool,
    pub purpose: String,
}

/// Dependency types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyType {
    QuantumLibrary,
    ClassicalLibrary,
    Algorithm,
    Data,
    Model,
    Hardware,
    Service,
}

/// Algorithm code representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmCode {
    pub primary_language: String,
    pub code_files: Vec<CodeFile>,
    pub entry_point: String,
    pub build_instructions: BuildInstructions,
    pub runtime_requirements: RuntimeRequirements,
}

/// Code file in the algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    pub filename: String,
    pub content: String,
    pub file_type: FileType,
    pub checksum: String,
    pub size_bytes: usize,
}

/// File types in algorithm packages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FileType {
    Source,
    Header,
    Configuration,
    Data,
    Documentation,
    Test,
    Build,
    Resource,
}

/// Build instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInstructions {
    pub build_system: String,
    pub build_commands: Vec<String>,
    pub environment_setup: Vec<String>,
    pub dependencies_install: Vec<String>,
    pub validation_commands: Vec<String>,
}

/// Runtime requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    pub python_version: Option<String>,
    pub required_packages: Vec<String>,
    pub environment_variables: HashMap<String, String>,
    pub resource_constraints: ResourceConstraints,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_mb: usize,
    pub max_cpu_cores: usize,
    pub max_execution_time_seconds: usize,
    pub max_quantum_volume: f64,
    pub max_network_bandwidth_mbps: f64,
}

/// Algorithm documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDocumentation {
    pub readme: String,
    pub api_documentation: String,
    pub theory_background: String,
    pub usage_examples: Vec<UsageExample>,
    pub tutorials: Vec<Tutorial>,
    pub faq: Vec<FAQEntry>,
    pub citations: Vec<Citation>,
    pub changelog: String,
}

/// Usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    pub title: String,
    pub description: String,
    pub code: String,
    pub expected_output: String,
    pub complexity: ExampleComplexity,
}

/// Example complexity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExampleComplexity {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Tutorial information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tutorial {
    pub title: String,
    pub description: String,
    pub duration_minutes: usize,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
    pub content: String,
    pub exercises: Vec<Exercise>,
}

/// Tutorial exercise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exercise {
    pub question: String,
    pub hint: Option<String>,
    pub solution: String,
    pub difficulty: ExampleComplexity,
}

/// FAQ entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FAQEntry {
    pub question: String,
    pub answer: String,
    pub category: String,
    pub votes: i32,
}

/// Citation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub title: String,
    pub authors: Vec<String>,
    pub journal: Option<String>,
    pub year: u16,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub url: Option<String>,
}

/// Algorithm test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTestSuite {
    pub unit_tests: Vec<UnitTest>,
    pub integration_tests: Vec<IntegrationTest>,
    pub performance_tests: Vec<PerformanceTest>,
    pub correctness_tests: Vec<CorrectnessTest>,
    pub regression_tests: Vec<RegressionTest>,
    pub coverage_report: CoverageReport,
}

/// Unit test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitTest {
    pub test_name: String,
    pub description: String,
    pub test_code: String,
    pub expected_result: TestResult,
    pub timeout_seconds: u32,
}

/// Integration test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTest {
    pub test_name: String,
    pub description: String,
    pub test_scenario: String,
    pub platforms_tested: Vec<String>,
    pub expected_behavior: String,
}

/// Performance test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTest {
    pub test_name: String,
    pub metrics: Vec<PerformanceMetric>,
    pub baseline_values: HashMap<String, f64>,
    pub acceptance_criteria: HashMap<String, f64>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub metric_name: String,
    pub metric_type: MetricType,
    pub unit: String,
    pub description: String,
}

/// Metric types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    ExecutionTime,
    MemoryUsage,
    QuantumVolume,
    CircuitDepth,
    GateCount,
    Fidelity,
    SuccessProbability,
    ErrorRate,
    Throughput,
    Custom(String),
}

/// Correctness test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessTest {
    pub test_name: String,
    pub input_specification: String,
    pub expected_output: String,
    pub verification_method: VerificationMethod,
    pub tolerance: Option<f64>,
}

/// Verification methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationMethod {
    ExactComparison,
    NumericalComparison,
    StatisticalTest,
    PropertyVerification,
    CrossValidation,
    Custom(String),
}

/// Regression test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTest {
    pub test_name: String,
    pub previous_version: String,
    pub comparison_metrics: Vec<String>,
    pub acceptable_regression: f64,
}

/// Test result types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestResult {
    Pass,
    Fail,
    Skip,
    Error,
    Timeout,
}

/// Coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub line_coverage: f64,
    pub function_coverage: f64,
    pub branch_coverage: f64,
    pub quantum_gate_coverage: f64,
    pub detailed_report: String,
}

/// Performance benchmark data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    pub benchmark_id: String,
    pub platform: String,
    pub timestamp: SystemTime,
    pub problem_size: usize,
    pub metrics: HashMap<String, f64>,
    pub environment_info: EnvironmentInfo,
    pub verification_status: VerificationStatus,
}

/// Environment information for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub platform_name: String,
    pub platform_version: String,
    pub hardware_specs: HashMap<String, String>,
    pub software_versions: HashMap<String, String>,
    pub configuration: HashMap<String, String>,
}

/// Verification status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    Unverified,
    Failed,
    Pending,
}

/// Licensing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicensingInfo {
    pub license_type: LicenseType,
    pub license_text: String,
    pub commercial_use_allowed: bool,
    pub attribution_required: bool,
    pub modification_allowed: bool,
    pub redistribution_allowed: bool,
    pub patent_grant: bool,
    pub copyleft: bool,
}

/// License types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LicenseType {
    MIT,
    Apache2,
    GPL3,
    BSD3Clause,
    Creative,
    Proprietary,
    Academic,
    Custom(String),
}

/// Registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationInfo {
    pub registered_at: SystemTime,
    pub last_updated: SystemTime,
    pub registration_status: RegistrationStatus,
    pub review_status: ReviewStatus,
    pub moderator_notes: Vec<ModeratorNote>,
    pub verification_badges: Vec<VerificationBadge>,
}

/// Registration status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegistrationStatus {
    Draft,
    Submitted,
    UnderReview,
    Approved,
    Rejected,
    Deprecated,
    Archived,
}

/// Review status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewStatus {
    Pending,
    InProgress,
    Completed,
    RequiresChanges,
    Approved,
    Rejected,
}

/// Moderator note
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeratorNote {
    pub moderator_id: String,
    pub timestamp: SystemTime,
    pub note_type: NoteType,
    pub content: String,
    pub visibility: NoteVisibility,
}

/// Note types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoteType {
    Review,
    Approval,
    Rejection,
    Improvement,
    Warning,
    Information,
}

/// Note visibility levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoteVisibility {
    Public,
    AuthorOnly,
    ModeratorsOnly,
    Internal,
}

/// Verification badges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationBadge {
    pub badge_type: BadgeType,
    pub awarded_by: String,
    pub awarded_at: SystemTime,
    pub criteria_met: Vec<String>,
    pub valid_until: Option<SystemTime>,
}

/// Badge types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BadgeType {
    Verified,
    HighPerformance,
    WellDocumented,
    Tested,
    Educational,
    Research,
    Production,
    Innovative,
    Popular,
    Maintained,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub total_downloads: u64,
    pub total_deployments: u64,
    pub unique_users: u64,
    pub average_rating: f64,
    pub total_ratings: u64,
    pub success_rate: f64,
    pub last_30_days: UsageStats,
    pub historical_data: Vec<HistoricalUsage>,
}

/// Usage statistics for specific periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub downloads: u64,
    pub deployments: u64,
    pub unique_users: u64,
    pub ratings: u64,
    pub average_rating: f64,
}

/// Historical usage data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalUsage {
    pub date: SystemTime,
    pub stats: UsageStats,
}

/// Search index for algorithm discovery
#[derive(Debug, Clone)]
pub struct SearchIndex {
    keyword_index: HashMap<String, HashSet<String>>,
    category_index: HashMap<AlgorithmCategory, HashSet<String>>,
    tag_index: HashMap<String, HashSet<String>>,
    author_index: HashMap<String, HashSet<String>>,
    performance_index: BTreeMap<String, Vec<(String, f64)>>,
}

/// Algorithm performance data
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceData {
    pub algorithm_id: String,
    pub benchmarks: Vec<PerformanceBenchmark>,
    pub average_performance: HashMap<String, f64>,
    pub performance_trend: PerformanceTrend,
    pub comparison_data: HashMap<String, f64>,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub trending_direction: TrendDirection,
    pub trend_strength: f64,
    pub last_updated: SystemTime,
    pub significant_changes: Vec<PerformanceChange>,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// Performance change record
#[derive(Debug, Clone)]
pub struct PerformanceChange {
    pub metric: String,
    pub change_percentage: f64,
    pub change_timestamp: SystemTime,
    pub context: String,
}

/// Algorithm registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRegistration {
    pub metadata: AlgorithmMetadata,
    pub code: AlgorithmCode,
    pub documentation: AlgorithmDocumentation,
    pub test_suite: AlgorithmTestSuite,
    pub licensing: LicensingInfo,
}

impl AlgorithmRegistry {
    /// Create a new algorithm registry
    pub fn new(config: &RegistryConfig) -> DeviceResult<Self> {
        Ok(Self {
            config: config.clone(),
            algorithms: HashMap::new(),
            categories: HashMap::new(),
            tags: HashMap::new(),
            search_index: SearchIndex::new(),
            algorithm_dependencies: HashMap::new(),
            algorithm_performance: HashMap::new(),
        })
    }

    /// Initialize the registry
    pub async fn initialize(&self) -> DeviceResult<()> {
        // Initialize search indices and data structures
        Ok(())
    }

    /// Register a new algorithm
    pub async fn register_algorithm(
        &mut self,
        registration: AlgorithmRegistration,
    ) -> DeviceResult<String> {
        let algorithm_id = Uuid::new_v4().to_string();

        // Validate algorithm size
        let total_size = self.calculate_algorithm_size(&registration.code)?;
        if total_size > self.config.max_algorithm_size {
            return Err(DeviceError::InvalidInput(format!(
                "Algorithm size {} exceeds maximum allowed size {}",
                total_size, self.config.max_algorithm_size
            )));
        }

        // Check algorithm limit
        if self.algorithms.len() >= self.config.max_algorithms {
            return Err(DeviceError::ResourceExhaustion(
                "Maximum number of algorithms reached".to_string(),
            ));
        }

        // Create registered algorithm
        let registered_algorithm = RegisteredAlgorithm {
            algorithm_id: algorithm_id.clone(),
            metadata: registration.metadata.clone(),
            code: registration.code,
            documentation: registration.documentation,
            test_suite: registration.test_suite,
            performance_benchmarks: vec![],
            licensing: registration.licensing,
            registration_info: RegistrationInfo {
                registered_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                registration_status: RegistrationStatus::Submitted,
                review_status: ReviewStatus::Pending,
                moderator_notes: vec![],
                verification_badges: vec![],
            },
            usage_statistics: UsageStatistics::default(),
        };

        // Update indices
        self.update_search_index(&algorithm_id, &registration.metadata);
        self.update_category_index(&algorithm_id, &registration.metadata.category);
        self.update_tag_index(&algorithm_id, &registration.metadata.tags);

        // Store algorithm
        self.algorithms
            .insert(algorithm_id.clone(), registered_algorithm);

        Ok(algorithm_id)
    }

    /// Get algorithm by ID
    pub async fn get_algorithm(
        &self,
        algorithm_id: &str,
    ) -> DeviceResult<Option<RegisteredAlgorithm>> {
        Ok(self.algorithms.get(algorithm_id).cloned())
    }

    /// Get algorithm count
    pub async fn get_algorithm_count(&self) -> DeviceResult<usize> {
        Ok(self.algorithms.len())
    }

    /// Search algorithms by keyword
    pub async fn search_by_keyword(&self, keyword: &str) -> DeviceResult<Vec<String>> {
        if let Some(algorithm_ids) = self.search_index.keyword_index.get(keyword) {
            Ok(algorithm_ids.iter().cloned().collect())
        } else {
            Ok(vec![])
        }
    }

    /// Get algorithms by category
    pub async fn get_by_category(&self, category: &AlgorithmCategory) -> DeviceResult<Vec<String>> {
        if let Some(algorithm_ids) = self.search_index.category_index.get(category) {
            Ok(algorithm_ids.iter().cloned().collect())
        } else {
            Ok(vec![])
        }
    }

    /// Update algorithm performance data
    pub async fn update_performance_data(
        &mut self,
        algorithm_id: &str,
        benchmark: PerformanceBenchmark,
    ) -> DeviceResult<()> {
        if let Some(algorithm) = self.algorithms.get_mut(algorithm_id) {
            algorithm.performance_benchmarks.push(benchmark.clone());

            // Update performance tracking
            if let Some(perf_data) = self.algorithm_performance.get_mut(algorithm_id) {
                perf_data.benchmarks.push(benchmark);
            } else {
                let perf_data = AlgorithmPerformanceData {
                    algorithm_id: algorithm_id.to_string(),
                    benchmarks: vec![benchmark],
                    average_performance: HashMap::new(),
                    performance_trend: PerformanceTrend {
                        trending_direction: TrendDirection::Stable,
                        trend_strength: 0.0,
                        last_updated: SystemTime::now(),
                        significant_changes: vec![],
                    },
                    comparison_data: HashMap::new(),
                };
                self.algorithm_performance
                    .insert(algorithm_id.to_string(), perf_data);
            }
        }
        Ok(())
    }

    // Helper methods
    fn calculate_algorithm_size(&self, code: &AlgorithmCode) -> DeviceResult<usize> {
        Ok(code.code_files.iter().map(|f| f.size_bytes).sum())
    }

    fn update_search_index(&mut self, algorithm_id: &str, metadata: &AlgorithmMetadata) {
        // Update keyword index
        for keyword in &metadata.keywords {
            self.search_index
                .keyword_index
                .entry(keyword.clone())
                .or_insert_with(HashSet::new)
                .insert(algorithm_id.to_string());
        }

        // Update tag index
        for tag in &metadata.tags {
            self.search_index
                .tag_index
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(algorithm_id.to_string());
        }

        // Update author index
        self.search_index
            .author_index
            .entry(metadata.author.clone())
            .or_insert_with(HashSet::new)
            .insert(algorithm_id.to_string());
    }

    fn update_category_index(&mut self, algorithm_id: &str, category: &AlgorithmCategory) {
        self.search_index
            .category_index
            .entry(category.clone())
            .or_insert_with(HashSet::new)
            .insert(algorithm_id.to_string());
    }

    fn update_tag_index(&mut self, algorithm_id: &str, tags: &[String]) {
        for tag in tags {
            self.tags
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(algorithm_id.to_string());
        }
    }
}

impl SearchIndex {
    fn new() -> Self {
        Self {
            keyword_index: HashMap::new(),
            category_index: HashMap::new(),
            tag_index: HashMap::new(),
            author_index: HashMap::new(),
            performance_index: BTreeMap::new(),
        }
    }
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            total_downloads: 0,
            total_deployments: 0,
            unique_users: 0,
            average_rating: 0.0,
            total_ratings: 0,
            success_rate: 0.0,
            last_30_days: UsageStats::default(),
            historical_data: vec![],
        }
    }
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            downloads: 0,
            deployments: 0,
            unique_users: 0,
            ratings: 0,
            average_rating: 0.0,
        }
    }
}
