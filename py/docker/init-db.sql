-- QuantRS2 Database Initialization Script

-- Create database schema for quantum experiments
CREATE SCHEMA IF NOT EXISTS quantum_experiments;

-- Table for storing quantum circuit experiments
CREATE TABLE IF NOT EXISTS quantum_experiments.circuits (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    circuit_data JSONB NOT NULL,
    n_qubits INTEGER NOT NULL,
    depth INTEGER,
    gate_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Table for storing experiment results
CREATE TABLE IF NOT EXISTS quantum_experiments.results (
    id SERIAL PRIMARY KEY,
    circuit_id INTEGER REFERENCES quantum_experiments.circuits(id) ON DELETE CASCADE,
    execution_time FLOAT NOT NULL,
    memory_usage BIGINT,
    shots INTEGER DEFAULT 1024,
    backend VARCHAR(255),
    result_data JSONB NOT NULL,
    error_rate FLOAT,
    fidelity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    environment_info JSONB DEFAULT '{}'::jsonb
);

-- Table for performance benchmarks
CREATE TABLE IF NOT EXISTS quantum_experiments.benchmarks (
    id SERIAL PRIMARY KEY,
    benchmark_name VARCHAR(255) NOT NULL,
    benchmark_type VARCHAR(100) NOT NULL,
    execution_time FLOAT NOT NULL,
    memory_usage BIGINT,
    additional_metrics JSONB DEFAULT '{}'::jsonb,
    environment_info JSONB DEFAULT '{}'::jsonb,
    git_commit_hash VARCHAR(40),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(benchmark_name, git_commit_hash, created_at)
);

-- Table for user sessions (if needed)
CREATE TABLE IF NOT EXISTS quantum_experiments.user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_circuits_name ON quantum_experiments.circuits(name);
CREATE INDEX IF NOT EXISTS idx_circuits_tags ON quantum_experiments.circuits USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_circuits_created_at ON quantum_experiments.circuits(created_at);
CREATE INDEX IF NOT EXISTS idx_results_circuit_id ON quantum_experiments.results(circuit_id);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON quantum_experiments.results(created_at);
CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON quantum_experiments.benchmarks(benchmark_name);
CREATE INDEX IF NOT EXISTS idx_benchmarks_created_at ON quantum_experiments.benchmarks(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON quantum_experiments.user_sessions(expires_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for circuits table
CREATE TRIGGER update_circuits_updated_at BEFORE UPDATE ON quantum_experiments.circuits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data for demonstration
INSERT INTO quantum_experiments.circuits (name, description, circuit_data, n_qubits, depth, gate_count, tags) VALUES
('Bell State', 'Simple Bell state preparation circuit', '{"gates": [{"type": "H", "qubit": 0}, {"type": "CNOT", "control": 0, "target": 1}]}', 2, 2, 2, ARRAY['entanglement', 'basic']),
('GHZ State', '3-qubit GHZ state preparation', '{"gates": [{"type": "H", "qubit": 0}, {"type": "CNOT", "control": 0, "target": 1}, {"type": "CNOT", "control": 1, "target": 2}]}', 3, 3, 3, ARRAY['entanglement', 'multiparticle']),
('Quantum Fourier Transform', '3-qubit QFT implementation', '{"gates": [{"type": "H", "qubit": 0}, {"type": "CPHASE", "control": 1, "target": 0, "angle": 1.5708}, {"type": "H", "qubit": 1}, {"type": "CPHASE", "control": 2, "target": 0, "angle": 0.7854}, {"type": "CPHASE", "control": 2, "target": 1, "angle": 1.5708}, {"type": "H", "qubit": 2}, {"type": "SWAP", "qubit1": 0, "qubit2": 2}]}', 3, 7, 7, ARRAY['fourier', 'algorithm']);

-- Grant permissions to quantrs2 user
GRANT ALL PRIVILEGES ON SCHEMA quantum_experiments TO quantrs2;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantum_experiments TO quantrs2;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA quantum_experiments TO quantrs2;