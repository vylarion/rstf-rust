use anyhow::{anyhow, Context, Result};
use argon2::Argon2;
use chacha20poly1305::aead::stream::{
    DecryptorBE32, EncryptorBE32, Nonce as NonceStream, StreamBE32,
};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit};
use clap::{Parser, Subcommand};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::min;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use zeroize::Zeroize;
use zstd::stream::read::Decoder as ZstdDecoder;
use zstd::stream::write::Encoder as ZstdEncoder;

type TipeNonce = NonceStream<ChaCha20Poly1305, StreamBE32<ChaCha20Poly1305>>;

const CHUNK_SIZE: usize = 64 * 1024;

#[derive(Serialize, Deserialize, Debug)]
struct RstfHeader {
    is_dir: bool,
    original_name: String,
    original_size: u64,
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Pack {
        input: PathBuf,
        #[arg(long)]
        wipe: bool,
        #[arg(long, default_value = "5")]
        level: i32,
        #[arg(long, short = 'k')]
        keyfile: Option<PathBuf>,
    },
    Unpack {
        input: PathBuf,
        #[arg(long, short = 'k')]
        keyfile: Option<PathBuf>,
    },
    List {
        input: PathBuf,
        #[arg(long, short = 'k')]
        keyfile: Option<PathBuf>,
    },
}

// Credential Processing Helper
fn process_credentials(
    salt: &[u8],
    keyfile_path: Option<PathBuf>,
    confirm_password: bool,
) -> Result<[u8; 32]> {
    let mut password =
        rpassword::prompt_password("Enter password: ").context("Failed to read password")?;

    if confirm_password {
        let confirm = rpassword::prompt_password("Confirm password: ")
            .context("Failed to read password confirmation")?;
        if password != confirm {
            password.zeroize();
            anyhow::bail!("Passwords do not match. Aborting.");
        }
    }

    let mut combined_credentials = password.as_bytes().to_vec();

    if let Some(path) = keyfile_path {
        println!("Reading keyfile: {}", path.display());
        let mut file = File::open(path).context("Failed to open keyfile")?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher).context("Failed to read keyfile")?;
        let hash = hasher.finalize();
        combined_credentials.extend_from_slice(&hash);
    }

    let argon2 = Argon2::default();
    let mut key = [0u8; 32];
    argon2
        .hash_password_into(&combined_credentials, salt, &mut key)
        .map_err(|_| anyhow!("Key derivation failed"))?;

    password.zeroize();
    combined_credentials.zeroize();

    Ok(key)
}

fn secure_wipe_file(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for: {}", path.display()))?;
    let size = metadata.len();
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!("Failed to open for wiping: {}", path.display()))?;

    let mut rng = rand::thread_rng();
    let mut buf = [0u8; 64 * 1024];
    let mut remaining = size;
    while remaining > 0 {
        let to_write = min(remaining as usize, buf.len());
        rng.fill(&mut buf[..to_write]);
        file.write_all(&buf[..to_write])?;
        remaining -= to_write as u64;
    }
    file.sync_all()?;
    drop(file);

    fs::remove_file(path)
        .with_context(|| format!("Failed to remove after wiping: {}", path.display()))?;
    Ok(())
}

fn secure_wipe_path(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for: {}", path.display()))?;

    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            secure_wipe_path(&entry.path())?;
        }
        fs::remove_dir(path)
            .with_context(|| format!("Failed to remove directory: {}", path.display()))?;
    } else {
        secure_wipe_file(path)?;
    }
    Ok(())
}

struct EncryptedWriter<W: Write> {
    inner: W,
    encryptor: Option<EncryptorBE32<ChaCha20Poly1305>>,
    buffer: Vec<u8>,
    finalized: bool,
}

// EncryptedWriter Implementation
impl<W: Write> EncryptedWriter<W> {
    fn new(inner: W, encryptor: EncryptorBE32<ChaCha20Poly1305>) -> Self {
        Self {
            inner,
            encryptor: Some(encryptor),
            buffer: Vec::with_capacity(CHUNK_SIZE),
            finalized: false,
        }
    }

    fn flush_intermediate(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let encryptor = self.encryptor.as_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "Stream already finalized")
        })?;

        let ciphertext = encryptor
            .encrypt_next(self.buffer.as_slice())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Encryption failed"))?;

        self.inner.write_all(&ciphertext)?;
        self.buffer.clear();
        Ok(())
    }

    fn finalize(&mut self) -> std::io::Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        let encryptor = self.encryptor.take().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "Stream already finalized")
        })?;

        let ciphertext = encryptor
            .encrypt_last(self.buffer.as_slice())
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "Final encryption failed")
            })?;

        self.inner.write_all(&ciphertext)?;
        self.buffer.clear();
        self.inner.flush()?;
        Ok(())
    }
}

// Write Trait for EncryptedWriter
impl<W: Write> Write for EncryptedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.finalized {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Cannot write to finalized stream",
            ));
        }

        let mut total_written = 0;
        while total_written < buf.len() {
            let space_left = CHUNK_SIZE - self.buffer.len();
            let to_copy = min(space_left, buf.len() - total_written);
            self.buffer
                .extend_from_slice(&buf[total_written..total_written + to_copy]);
            total_written += to_copy;

            if self.buffer.len() == CHUNK_SIZE {
                self.flush_intermediate()?;
            }
        }
        Ok(total_written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

// Drop Trait for EncryptedWriter
impl<W: Write> Drop for EncryptedWriter<W> {
    fn drop(&mut self) {
        if !self.finalized {
            let _ = self.finalize();
        }
    }
}

struct DecryptedReader<R: Read> {
    inner: R,
    decryptor: Option<DecryptorBE32<ChaCha20Poly1305>>,
    plaintext_buffer: Vec<u8>,
    offset: usize,
    eof: bool,
    staged_ciphertext: Option<Vec<u8>>,
}

// DecryptedReader Implementation
impl<R: Read> DecryptedReader<R> {
    fn new(inner: R, decryptor: DecryptorBE32<ChaCha20Poly1305>) -> Self {
        Self {
            inner,
            decryptor: Some(decryptor),
            plaintext_buffer: Vec::new(),
            offset: 0,
            eof: false,
            staged_ciphertext: None,
        }
    }

    fn read_encrypted_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        let encrypted_chunk_size = CHUNK_SIZE + 16;
        let mut buf = vec![0u8; encrypted_chunk_size];

        let mut read_bytes = 0;
        while read_bytes < encrypted_chunk_size {
            match self.inner.read(&mut buf[read_bytes..]) {
                Ok(0) => break,
                Ok(n) => read_bytes += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }

        if read_bytes == 0 {
            Ok(None)
        } else {
            buf.truncate(read_bytes);
            Ok(Some(buf))
        }
    }

    fn decrypt_next_chunk(&mut self) -> std::io::Result<bool> {
        if self.staged_ciphertext.is_none() {
            self.staged_ciphertext = self.read_encrypted_chunk()?;
            if self.staged_ciphertext.is_none() {
                self.eof = true;
                return Ok(false);
            }
        }

        let next_chunk = self.read_encrypted_chunk()?;
        let staged = self.staged_ciphertext.take().unwrap();

        let plaintext = if next_chunk.is_none() {
            self.eof = true;
            let decryptor = self.decryptor.take().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::Other, "Decryptor already consumed")
            })?;
            decryptor.decrypt_last(staged.as_slice()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Decryption failed (MAC Error)",
                )
            })?
        } else {
            let decryptor = self.decryptor.as_mut().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::Other, "Decryptor already consumed")
            })?;
            let result = decryptor.decrypt_next(staged.as_slice()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Decryption failed (MAC Error)",
                )
            })?;
            self.staged_ciphertext = next_chunk;
            result
        };

        self.plaintext_buffer = plaintext;
        self.offset = 0;
        Ok(true)
    }
}

// Read Trait for DecryptedReader
impl<R: Read> Read for DecryptedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.offset >= self.plaintext_buffer.len() {
            if self.eof {
                return Ok(0);
            }
            if !self.decrypt_next_chunk()? {
                return Ok(0);
            }
        }

        let available = self.plaintext_buffer.len() - self.offset;
        let to_copy = min(available, buf.len());
        buf[..to_copy]
            .copy_from_slice(&self.plaintext_buffer[self.offset..self.offset + to_copy]);
        self.offset += to_copy;

        Ok(to_copy)
    }
}

// Main Entry Point
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Pack {
            input,
            wipe,
            level,
            keyfile,
        } => pack(input, wipe, level, keyfile),
        Commands::Unpack { input, keyfile } => unpack(input, keyfile),
        Commands::List { input, keyfile } => list(input, keyfile),
    }
}

// Pack Function
fn pack(input_path: PathBuf, wipe: bool, level: i32, keyfile: Option<PathBuf>) -> Result<()> {
    let salt: [u8; 16] = rand::thread_rng().gen();

    let key = process_credentials(&salt, keyfile, true)?;

    let mut output_path = input_path.clone();
    if let Some(name) = input_path.file_name() {
        let mut new_name = name.to_os_string();
        new_name.push(".rstf");
        output_path.set_file_name(new_name);
    } else {
        output_path.set_extension("rstf");
    }

    let output_file = File::create(&output_path).context("Failed to create output file")?;
    let mut writer = BufWriter::with_capacity(CHUNK_SIZE, output_file);

    let nonce: [u8; 7] = rand::thread_rng().gen();
    writer.write_all(&salt)?;
    writer.write_all(&nonce)?;

    let metadata = fs::metadata(&input_path).context("Failed to read metadata")?;
    let is_dir = metadata.is_dir();
    let total_size = if is_dir { 0 } else { metadata.len() };

    let header = RstfHeader {
        is_dir,
        original_name: input_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        original_size: total_size,
    };
    let header_bytes = bincode::serialize(&header)?;
    let header_len = header_bytes.len() as u32;

    let key_struct = chacha20poly1305::Key::from_slice(&key);
    let aead = ChaCha20Poly1305::new(key_struct);

    let s_nonce = TipeNonce::from_slice(&nonce);
    let encryptor = EncryptorBE32::from_aead(aead, s_nonce);

    let mut crypto_writer = EncryptedWriter::new(writer, encryptor);

    crypto_writer.write_all(&header_len.to_le_bytes())?;
    crypto_writer.write_all(&header_bytes)?;

    let mut zstd_writer = ZstdEncoder::new(crypto_writer, level)?;
    zstd_writer.multithread(num_cpus::get() as u32)?;

    println!("Packing {}...", input_path.display());
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));

    if is_dir {
        let mut tar_builder = tar::Builder::new(&mut zstd_writer);
        tar_builder.append_dir_all(&header.original_name, &input_path)?;
        tar_builder.finish()?;
        pb.finish_with_message("Directory packed");
    } else {
        let input_file = File::open(&input_path)?;
        let mut input_with_pb = pb.wrap_read(input_file);
        std::io::copy(&mut input_with_pb, &mut zstd_writer)?;
        pb.finish_with_message("File packed");
    }

    let mut crypto_writer = zstd_writer.finish()?;
    crypto_writer
        .finalize()
        .context("Failed to finalize encrypted stream")?;

    if wipe {
        print!(
            "\nSecurely wipe original '{}'? (y/N): ",
            input_path.display()
        );
        std::io::stdout().flush()?;

        let mut input_string = String::new();
        std::io::stdin()
            .read_line(&mut input_string)
            .context("Failed to read input")?;

        if input_string.trim().to_lowercase() == "y" {
            println!("Securely wiping (overwriting with random data)...");
            secure_wipe_path(&input_path)
                .context("Failed to securely wipe original data")?;
            println!("Original data securely wiped.");
        } else {
            println!("Wipe cancelled. Original data preserved.");
        }
    }

    Ok(())
}

// Unpack Function
fn unpack(input_path: PathBuf, keyfile: Option<PathBuf>) -> Result<()> {
    let mut input_file = File::open(&input_path).context("Failed to open .rstf")?;

    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 7];
    input_file.read_exact(&mut salt)?;
    input_file.read_exact(&mut nonce)?;

    let key = process_credentials(&salt, keyfile, false)?;

    let key_struct = chacha20poly1305::Key::from_slice(&key);
    let aead = ChaCha20Poly1305::new(key_struct);

    let s_nonce = TipeNonce::from_slice(&nonce);
    let decryptor = DecryptorBE32::from_aead(aead, s_nonce);

    let mut crypto_reader = DecryptedReader::new(input_file, decryptor);

    let mut len_bytes = [0u8; 4];
    crypto_reader
        .read_exact(&mut len_bytes)
        .context("Failed to decrypt header (Wrong password or Wrong Keyfile?)")?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;

    let mut header_data = vec![0u8; header_len];
    crypto_reader.read_exact(&mut header_data)?;
    let header: RstfHeader = bincode::deserialize(&header_data)?;

    println!("Unpacking: {}", header.original_name);

    let mut zstd_reader = ZstdDecoder::new(crypto_reader)?;

    let pb = ProgressBar::new(header.original_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes} ({bytes_per_sec})")?
        .progress_chars("#>-"));

    if header.is_dir {
        let mut archive = tar::Archive::new(&mut zstd_reader);
        archive.unpack(".").context("Failed to extract tar")?;
    } else {
        let output_file = File::create(&header.original_name)?;
        let mut output_with_pb = pb.wrap_write(output_file);
        std::io::copy(&mut zstd_reader, &mut output_with_pb)?;
    }

    pb.finish_with_message("Done!");
    Ok(())
}

// List Function
fn list(input_path: PathBuf, keyfile: Option<PathBuf>) -> Result<()> {
    let mut input_file = File::open(&input_path)?;
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 7];
    input_file.read_exact(&mut salt)?;
    input_file.read_exact(&mut nonce)?;

    let key = process_credentials(&salt, keyfile, false)?;

    let key_struct = chacha20poly1305::Key::from_slice(&key);
    let aead = ChaCha20Poly1305::new(key_struct);

    let s_nonce = TipeNonce::from_slice(&nonce);
    let decryptor = DecryptorBE32::from_aead(aead, s_nonce);
    let mut crypto_reader = DecryptedReader::new(input_file, decryptor);

    let mut len_bytes = [0u8; 4];
    crypto_reader
        .read_exact(&mut len_bytes)
        .context("Decrypt failed (Wrong password or keyfile?)")?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;
    let mut header_data = vec![0u8; header_len];
    crypto_reader.read_exact(&mut header_data)?;
    let header: RstfHeader = bincode::deserialize(&header_data)?;

    println!("\n[RSTF INFO]");
    println!("Name : {}", header.original_name);
    println!(
        "Type : {}",
        if header.is_dir { "Directory" } else { "File" }
    );
    println!("Size : {}", HumanBytes(header.original_size));

    Ok(())
}
