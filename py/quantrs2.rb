# Homebrew Formula for QuantRS2-Py
# This formula allows installation of QuantRS2 Python bindings via Homebrew
# Installation: brew install quantrs2.rb

class Quantrs2 < Formula
  include Language::Python::Virtualenv

  desc "High-performance quantum computing framework with Python bindings"
  homepage "https://github.com/cool-japan/quantrs"
  url "https://github.com/cool-japan/quantrs/archive/refs/tags/v0.1.0-beta.3.tar.gz"
  sha256 "" # TODO: Add SHA256 checksum after release
  license "MIT OR Apache-2.0"
  head "https://github.com/cool-japan/quantrs.git", branch: "master"

  depends_on "rust" => :build
  depends_on "python@3.10"
  depends_on "maturin" => :build
  depends_on "cmake" => :build

  # Optional dependencies for enhanced features
  depends_on "openblas" => :optional
  depends_on "cuda" => :optional

  def install
    # Set up Python environment
    venv = virtualenv_create(libexec, "python3.10")

    # Install maturin if not available
    system "pip3", "install", "maturin"

    # Build and install the Python package from the py directory
    cd "py" do
      # Build with maturin
      system "maturin", "build", "--release", "--strip"

      # Install the built wheel
      wheel = Dir["target/wheels/*.whl"].first
      system libexec/"bin/pip", "install", wheel
    end

    # Create wrapper scripts
    bin.install_symlink libexec/"bin/quantrs2" if File.exist?(libexec/"bin/quantrs2")
  end

  def caveats
    <<~EOS
      QuantRS2 has been installed!

      To use the Python bindings, you may need to add the installation directory
      to your PYTHONPATH:
        export PYTHONPATH="#{libexec}/lib/python3.10/site-packages:$PYTHONPATH"

      For GPU acceleration (CUDA), install with:
        brew install quantrs2 --with-cuda

      For enhanced linear algebra performance, install with:
        brew install quantrs2 --with-openblas

      Documentation: https://github.com/cool-japan/quantrs/tree/master/py
      Examples: https://github.com/cool-japan/quantrs/tree/master/py/examples
    EOS
  end

  test do
    system libexec/"bin/python", "-c", <<~EOS
      import quantrs2 as qr
      circuit = qr.PyCircuit(2)
      circuit.h(0)
      circuit.cnot(0, 1)
      result = circuit.run()
      probs = result.state_probabilities()
      assert len(probs) == 4
      print("QuantRS2 test passed!")
    EOS
  end
end
