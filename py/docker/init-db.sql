-- QuantRS2 Database Initialization Script
-- Creates comprehensive database schema for QuantRS2 services

-- Create extensions for enhanced functionality
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "hstore";

-- Create database schema for quantum experiments
CREATE SCHEMA IF NOT EXISTS quantum_experiments;
CREATE SCHEMA IF NOT EXISTS user_management;
CREATE SCHEMA IF NOT EXISTS analytics;

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

-- User management tables
CREATE TABLE IF NOT EXISTS user_management.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    full_name VARCHAR(255),
    organization VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE IF NOT EXISTS user_management.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_management.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    permissions TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Analytics tables
CREATE TABLE IF NOT EXISTS analytics.usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_management.users(id),
    session_id UUID,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

CREATE TABLE IF NOT EXISTS analytics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION,
    unit VARCHAR(50),
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create additional indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON user_management.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON user_management.users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON user_management.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_timestamp ON analytics.usage_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON analytics.performance_metrics(service_name);

-- Insert default users
INSERT INTO user_management.users (username, email, full_name, organization, preferences) 
VALUES 
    ('admin', 'admin@quantrs2.local', 'QuantRS2 Administrator', 'QuantRS2 Team', '{"theme": "dark", "notifications": true}'),
    ('demo', 'demo@quantrs2.local', 'Demo User', 'Demo Organization', '{"theme": "light", "notifications": false}')
ON CONFLICT (username) DO NOTHING;

-- Grant permissions to quantrs2 user
GRANT ALL PRIVILEGES ON SCHEMA quantum_experiments TO quantrs2;
GRANT ALL PRIVILEGES ON SCHEMA user_management TO quantrs2;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO quantrs2;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantum_experiments TO quantrs2;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA user_management TO quantrs2;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO quantrs2;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA quantum_experiments TO quantrs2;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA user_management TO quantrs2;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO quantrs2;