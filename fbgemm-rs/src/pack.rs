/// Block column size: 2 kernel column blocks × 8 floats per SIMD lane = 16.
/// This matches FBGEMM's layout for both AVX2 (2×8) and Neon (4×4) paths.
pub const BLOCK_COL_SIZE: usize = 16;

/// Default block row size along the K dimension.
const DEFAULT_BROW: usize = 512;

/// A pre-packed B matrix in blocked row-major format.
///
/// The packed layout groups elements into (brow × bcol) blocks for
/// cache-friendly access by the GEMM micro-kernels.
pub struct PackedMatrix {
    pub(crate) nrow: usize, // K dimension
    pub(crate) ncol: usize, // N dimension
    pub(crate) brow: usize,
    pub(crate) last_brow: usize,
    pub(crate) bcol: usize, // = BLOCK_COL_SIZE
    pub(crate) nbrow: usize,
    pub(crate) nbcol: usize,
    pub(crate) data: Vec<f32>,
}

unsafe impl Send for PackedMatrix {}
unsafe impl Sync for PackedMatrix {}

impl PackedMatrix {
    /// Pack a K×N row-major matrix.
    pub fn new(k: usize, n: usize, src: &[f32]) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");
        let mut m = Self::alloc(k, n);
        m.pack_from_src(false, 1.0, src);
        m
    }

    /// Pack a K×N matrix from column-major (transposed) storage.
    pub fn from_transposed(k: usize, n: usize, src: &[f32]) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");
        let mut m = Self::alloc(k, n);
        m.pack_from_src(true, 1.0, src);
        m
    }

    /// Pack with a scaling factor: packed = alpha * B.
    pub fn with_alpha(k: usize, n: usize, src: &[f32], alpha: f32) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");
        let mut m = Self::alloc(k, n);
        m.pack_from_src(false, alpha, src);
        m
    }

    pub fn k(&self) -> usize {
        self.nrow
    }

    pub fn n(&self) -> usize {
        self.ncol
    }

    pub fn block_row_size(&self) -> usize {
        self.brow
    }

    pub fn block_col_size(&self) -> usize {
        self.bcol
    }

    /// Get a pointer to element (r, c) in the packed layout.
    pub(crate) fn at(&self, r: usize, c: usize) -> *const f32 {
        unsafe { self.data.as_ptr().add(self.addr(r, c)) }
    }

    fn alloc(nrow: usize, ncol: usize) -> Self {
        let brow = DEFAULT_BROW;
        let bcol = BLOCK_COL_SIZE;
        let nbrow = (nrow + brow - 1) / brow;
        let last_brow = if nrow % brow == 0 { brow } else { nrow % brow };
        let nbcol = (ncol + bcol - 1) / bcol;
        let size = brow * nbrow * bcol * nbcol;
        Self {
            nrow,
            ncol,
            brow,
            last_brow,
            bcol,
            nbrow,
            nbcol,
            data: vec![0.0; size],
        }
    }

    fn addr(&self, r: usize, c: usize) -> usize {
        let block_row_id = r / self.brow;
        let brow_offset = block_row_id * self.nbcol * self.brow * self.bcol;
        let block_col_id = c / self.bcol;
        let rows_in_block = if block_row_id as isize != self.nbrow as isize - 1 {
            self.brow
        } else {
            self.last_brow
        };
        let bcol_offset = block_col_id * rows_in_block * self.bcol;
        let block_offset = brow_offset + bcol_offset;
        let inblock_offset = (r % self.brow) * self.bcol + (c % self.bcol);
        block_offset + inblock_offset
    }

    fn pack_from_src(&mut self, transposed: bool, alpha: f32, src: &[f32]) {
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let src_val = if transposed {
                    src[i + self.nrow * j]
                } else {
                    src[i * self.ncol + j]
                };
                let idx = self.addr(i, j);
                self.data[idx] = alpha * src_val;
            }
        }
    }
}
