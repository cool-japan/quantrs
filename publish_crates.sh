#!/bin/bash

# QuantRS2 crates.io publish script
# Publishes crates in dependency order

set -e  # Exit on error

echo "🚀 Publishing QuantRS2 crates to crates.io..."

# Function to publish a crate with retry logic
publish_crate() {
    local crate_dir=$1
    local crate_name=$2
    
    echo ""
    echo "📦 Publishing $crate_name..."
    cd "$crate_dir"
    
    # Try to publish, with retry logic
    local retries=3
    local wait_time=30
    
    for i in $(seq 1 $retries); do
        if cargo publish; then
            echo "✅ Successfully published $crate_name"
            cd ..
            sleep 5  # Wait a bit for crates.io to process
            return 0
        else
            if [ $i -lt $retries ]; then
                echo "⚠️  Failed to publish $crate_name, retrying in ${wait_time}s... (attempt $i/$retries)"
                sleep $wait_time
            else
                echo "❌ Failed to publish $crate_name after $retries attempts"
                cd ..
                return 1
            fi
        fi
    done
}

# 1. core (no dependencies)
publish_crate "core" "quantrs2-core"

# 2. circuit (depends on core)
publish_crate "circuit" "quantrs2-circuit"

# 3. anneal (depends on core)
publish_crate "anneal" "quantrs2-anneal"

# 4. sim (depends on core, circuit)
publish_crate "sim" "quantrs2-sim"

# 5. device (depends on core, circuit)
publish_crate "device" "quantrs2-device"

# 6. ml (depends on core, circuit, sim)
publish_crate "ml" "quantrs2-ml"

# 7. tytan (depends on core, anneal)
publish_crate "tytan" "quantrs2-tytan"

echo ""
echo "🎉 All crates published successfully!"