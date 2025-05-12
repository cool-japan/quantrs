/// A transparent wrapper around a qubit identifier
///
/// This provides type safety for qubit references while
/// maintaining zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct QubitId(pub u32);

impl QubitId {
    /// Create a new qubit identifier
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw identifier value
    #[inline]
    pub const fn id(&self) -> u32 {
        self.0
    }
}

impl From<u32> for QubitId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<QubitId> for u32 {
    #[inline]
    fn from(qubit: QubitId) -> Self {
        qubit.0
    }
}

impl From<usize> for QubitId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<QubitId> for usize {
    #[inline]
    fn from(qubit: QubitId) -> Self {
        qubit.0 as usize
    }
}

impl From<i32> for QubitId {
    #[inline]
    fn from(id: i32) -> Self {
        Self(id as u32)
    }
}

/// A collection of qubits for multi-qubit operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QubitSet {
    qubits: Vec<QubitId>,
}

impl QubitSet {
    /// Create a new empty qubit set
    pub fn new() -> Self {
        Self { qubits: Vec::new() }
    }

    /// Create a qubit set with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            qubits: Vec::with_capacity(capacity),
        }
    }

    /// Add a qubit to the set
    pub fn add(&mut self, qubit: impl Into<QubitId>) {
        self.qubits.push(qubit.into());
    }

    /// Get all qubits in the set
    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    /// Get the number of qubits in the set
    pub fn len(&self) -> usize {
        self.qubits.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.qubits.is_empty()
    }
}

impl Default for QubitSet {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<QubitId>> for QubitSet {
    fn from(qubits: Vec<QubitId>) -> Self {
        Self { qubits }
    }
}

impl<const N: usize> From<[QubitId; N]> for QubitSet {
    fn from(qubits: [QubitId; N]) -> Self {
        Self {
            qubits: qubits.to_vec(),
        }
    }
}

impl<const N: usize> From<[u32; N]> for QubitSet {
    fn from(ids: [u32; N]) -> Self {
        let qubits = ids.iter().map(|&id| QubitId::new(id)).collect();
        Self { qubits }
    }
}
