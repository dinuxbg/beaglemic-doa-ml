// SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Create a training data set from the raw microphone recordings.
// Run this tool by passing the raw microphone recordings. It would
// store the data sets as raw files placed in a directory structure
// named per the features.
//
// Storing a large number of small datasets into an HDF5 container is
// no efficient. Storing a large number of small files in a filesystem
// directory structure is orders of magnitude faster.
//
// TODO - remove the "Cisms" and switch to modern C++ paradigms and style.

#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <wordexp.h>

// Input (and output!) audio format.
const int NCHANNELS = 8;
const int BITS_PER_SAMPLE = 32;
const int SAMPLES_PER_SECOND = 24000;

// Parsing parameters
const float INITIAL_SKIP_S = 0.5;	// Recording sometimes starts with a glitch.
const float SILENCE_TRAINING_S = 1.0;	// A period which we know is silent.
const float VALID_SAMPLE_THRESHOLD = 1.1; // Threshold over maximum silence to consider a sample valid.
const float VALID_SAMPLES_PERCENT = 10;	// Minimum percentage of valid samples to consider a chunk valid.

// Neural Network's input parameters.
const int OUT_NSAMPLES = 512;		//Output audio chunk size to save.

// Dataset size, in number of S32LE words.
const size_t OUT_DATASET_NWORDS = OUT_NSAMPLES * NCHANNELS;

// TODO - control it from the command line!
const bool VERBOSE = true;

const int OUT_DROP_PERCENT = 95;	// Randomly drop this number of datasets. Useful if
					// the input raw data is too large.

namespace fs = std::filesystem;

//----------------------------------------------------------------------------
static void fatal(const std::string &s)
{
	std::cerr << "ERROR: " << s << std::endl;
	std::cerr << "   errno=" << errno << std::endl;
	std::abort();
}

// Helper class for access to a large file consisting of
// consecutive signed 32-bit little-endian integer values.
class s32le_buf_t {
public:
	~s32le_buf_t() {
		if (this->raw)
			munmap((void *)this->raw, this->len * sizeof(int32_t));
	}

	// TODO - hide these under a sane iterator/container/operator[] interface.
	const int32_t *raw;
	off_t len;

	static std::shared_ptr<s32le_buf_t> open(std::string fpath)
	{
		int fd = ::open(fpath.c_str(), O_RDONLY);
		if (fd < 0)
			fatal("failed to open file \"" + fpath + "\"");

		struct stat statbuf;
		int err = fstat(fd, &statbuf);
		if (err < 0)
			fatal("failed to fstat file \"" + fpath + "\"");
		off_t len = statbuf.st_size  - (statbuf.st_size % sizeof(int32_t));
		void *tmp = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
		if (tmp == MAP_FAILED)
			fatal("failed to mmap file \"" + fpath + "\"");
		madvise(tmp, len, MADV_SEQUENTIAL);
		auto o = new s32le_buf_t(static_cast<const int32_t *>(tmp), len / sizeof(int32_t));

		close(fd);

		return std::shared_ptr<s32le_buf_t>(o);
	}

private:
	// Force usage only through shared_ptr.
	s32le_buf_t() : raw(NULL), len(0) {}
	s32le_buf_t(const int32_t *p, off_t l) : raw(p), len(l) {
		if (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
			fatal("big endian hosts not yet supported");
	}
};

//----------------------------------------------------------------------------

// Base class for outputting datasets to a filesystem tree.
class base_output {
public:
	const fs::path srcpath;

	base_output(const fs::path &_srcpath, const fs::path &_outbase)
		: srcpath(_srcpath), outbase(_outbase)
	{
	}
	virtual ~base_output()
	{
	}

	// Save all the variants of the given raw audio chunk to file(s) on disk.
	// This is virtual in order to allow custom variant preprocessing
	// before the actual data save.
	virtual bool save_chunk(const int32_t arr[OUT_NSAMPLES * NCHANNELS], off_t chunk_i, bool is_silence) = 0;

protected:
	const fs::path outbase;

	// Useful utility function to save raw data to a file.
	void save_to_file(const fs::path &path,
			const int32_t *arr, off_t chunk_i)
	{
		int rnd = std::rand() % 100;
		if (rnd < OUT_DROP_PERCENT)
			return;
		// Let's use filename() instead of stem() for a more definitive record of the origin.
		const auto fname = this->srcpath.filename().string() + "_" + std::to_string(chunk_i);
		fs::create_directories(outbase / path);
		const fs::path dst = outbase / path / fname;
		std::fstream s {dst, s.binary | s.trunc | s.out};
		if (!s.is_open()) {
			fatal("Failed to open " + dst.string());
		}
		s.write(reinterpret_cast<const char *>(arr), sizeof(arr[0]) * OUT_DATASET_NWORDS);
	}
};

// Output silence datasets.
class silence_output : public base_output {
public:
	silence_output(const fs::path &_srcpath, const fs::path &_outbase)
		: base_output(_srcpath, _outbase)
	{
	}
	virtual ~silence_output()
	{
	}
	virtual bool save_chunk(const int32_t *arr, off_t chunk_i, bool is_silence)
	{
		if (is_silence) {
			/* Doesn't matter.  We want to record the silence. */;
		}
		this->save_to_file("silence", arr, chunk_i);
		return true;
	}
};

// Output speech datasets from a particular angle.
class dataset_output : public base_output {
public:
	dataset_output(const fs::path &_srcpath, const fs::path &_outbase)
		: base_output(_srcpath, _outbase),
		  subangle(-1.0), elev(-1.0), distance(-1.0)
	{
 		/*
		 * Extract the physical environment settings for a
		 * microphone recording, given its filename.
		 *
		 * Example input: output-05.625deg-0elev-1.0m.raw
		 * Example parameters: ('05.625', '0', '1.0')
		 */
		int i_elev = 0;
		int n = std::sscanf(srcpath.filename().string().c_str(),
				"output-%fdeg-%delev-%fm.raw",
				&this->subangle, &i_elev, &this->distance);
		this->elev = i_elev;

		if (n != 3) {
			fatal(srcpath.filename().string() + " has invalid filename.");
		}

		// Initialize the angle directory paths, so they
		// can be easily reused when saving the chunks.
		for (int mic_offs = 0; mic_offs < NCHANNELS; mic_offs++) {
			float angle = this->subangle + mic_offs * (360.0 / NCHANNELS);
			char a_str[16], e_str[16], d_str[16];
			sprintf(a_str, "%1.3f", angle);
			sprintf(e_str, "%1.1f", this->elev);
			sprintf(d_str, "%1.1f", this->distance);
			fs::path path = a_str;
			path = path / e_str / d_str;
			this->angle_dirs[mic_offs] = path;
			//std::cout << "Directories: " << path << std::endl;
		}
	}
	virtual ~dataset_output()
	{
	}

	virtual bool save_chunk(const int32_t *arr, off_t chunk_i, bool is_silence)
	{
		// Don't record silence.
		if (is_silence)
			return false;

		for (int mic_offs = 0; mic_offs < NCHANNELS; mic_offs++) {
			int32_t data[OUT_DATASET_NWORDS];
			// This is important!!!!
			//
			// "Rotate" the emitting point per mic_offs.
			//
			// The raw input is recorded only for an angle
			// between MIC0 and MIC1. Here we simulate the full
			// range of angles between MIC0 and MIC7 by
			// "shifting" the channels.
			//
			// For example, for one recording of angle 5.6°, here
			// we output eight datasets for angles:
			//    5.6°   =  5.6° + 0 * (360° / 8)
			//    50.6°  =  5.6° + 1 * (360° / 8)
			//    95.6°  =  5.6° + 2 * (360° / 8)
			//    ....
			//    320.6° =  5.6° + 7 * (360° / 8)
			for (size_t si = 0; si < OUT_DATASET_NWORDS; si += NCHANNELS)
				for (size_t chi = 0; chi < NCHANNELS; chi++)
					data[si + (chi + mic_offs) % NCHANNELS] = arr[si + chi];
			// "Normalize" data by recording only the difference
			// from channel 0.
			//
			// Leave the raw PCM data for channel 0 itself. This data
			// is needed by the NN to detect silence.
			for (size_t si = 0; si < OUT_DATASET_NWORDS; si += NCHANNELS)
				for (size_t chi = 1; chi < NCHANNELS; chi++)
					data[si + chi] -= data[si];
			this->save_to_file(this->angle_dirs[mic_offs], data, chunk_i);
		}
		return true;
	}
private:
	float subangle;
	float elev;
	float distance;
	fs::path angle_dirs[NCHANNELS];
};
//----------------------------------------------------------------------------

// Calculate offset (in number of S32LE words) out of the given
// length of audio, in seconds.
static inline off_t secs2offs(double nsecs)
{
	double nsamples = double(SAMPLES_PER_SECOND) * nsecs;

	return off_t(std::floor(nsamples)) * NCHANNELS;
}

static bool int32_cmp_abs(int32_t a, int32_t b)
{
	return std::labs(a) < std::labs(b);
}

/*
 Parse a raw micriphone recording file. Detect chunks (intervals) of audio
 which are suitable for a training data set, and store them.

 Phases:
   1. Skip the glitch.
     The microphone recordings start with a glitch (loud noise), as an
     artifact of the processing running on the USB microphones.
   2. Train for silence.
     The first 2 seconds are silence (zeros output to the speaker).
     We use this time to detect the maximum amplitude, and record it
     as the noise threshold marker. The time is less than 2 seconds
     due to the glitch above.
   3. Chunk detection.
     Scan the rest of the file. If each successive chunk has
     enough samples above the threshold of silence, record
     them as useful for training.
*/
static void process_raw_audio_file(base_output &out)
{
	const std::string fpath = out.srcpath.string();

	std::cout << "Processing " << fpath << " ..." << std::endl;

	auto m = s32le_buf_t::open(fpath);

	const auto silence_scan_i = secs2offs(INITIAL_SKIP_S);
	const auto data_scan_i = silence_scan_i + secs2offs(SILENCE_TRAINING_S);

	if (silence_scan_i >= m->len || data_scan_i >= m->len)
		fatal("input file \"" + fpath + "\" is too short");

	const int32_t silence_max = std::labs(*std::max_element(m->raw + silence_scan_i, m->raw + data_scan_i, int32_cmp_abs));

	const int32_t silence_threshold = double(silence_max) * VALID_SAMPLE_THRESHOLD;

	const off_t chunk_len = OUT_NSAMPLES * NCHANNELS;
	const int nvals_threshold = double(chunk_len) * VALID_SAMPLES_PERCENT / 100.0;

	if (VERBOSE) {
		std::cout << "    Max silence sample: 0x" << std::hex << silence_max << std::endl;
		std::cout << std::dec;
		std::cout << "    Silence index: " << silence_scan_i << std::endl;
		std::cout << "    Data scan index: " << data_scan_i << std::endl;
		std::cout << "    Silence threshold: " << silence_max << std::endl;
		std::cout << "    Num values threshold: " << nvals_threshold;
		std::cout << "/" << chunk_len << std::endl;
	}

	int num_chunks = 0;

	for (off_t chunk_i = data_scan_i;
	     chunk_i <= (m->len - chunk_len);
	     chunk_i += chunk_len) {
		auto chunk = &m->raw[chunk_i];

		auto cmp_to_threshold = [silence_threshold](const int32_t val) {
			return std::labs(val) >= silence_threshold;
		};

		int nvals = std::count_if (chunk,
					   chunk + chunk_len,
					   cmp_to_threshold);

		const bool is_silence = (nvals >= nvals_threshold);

		if (out.save_chunk(chunk, chunk_i, is_silence))
			num_chunks++;
	}
	if (VERBOSE) {
		std::cout << "    Number of data chunks recorded: " << num_chunks;
		std::cout << " (" << ((num_chunks * chunk_len * 100) / m->len) << "%)" << std::endl;
	}
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	// TODO - proper argv parsing
	if (argc != 3)
		fatal("Usage: create-hdf5-data <RAW_AUDIO_DIRECTORY> <OUTPUT_HDF5>");

	const std::string fpattern = std::string(argv[1]) + "/output-*deg-*elev-*m.raw";
	const std::string fpattern_silence = std::string(argv[1]) + "/output-silence*.raw";

	const std::string output_directory = argv[2];

	wordexp_t exp;
	int st;

	if (1) {
		// Let's gamble :)
		auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
		auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
		std::srand(ns);
	} else {
		// If stable output is desired.
		std::srand(42);
	}

	st = wordexp(fpattern_silence.c_str(), &exp, WRDE_NOCMD | WRDE_SHOWERR | WRDE_UNDEF);
	if (st < 0)
		fatal("wordexp error");
	for (size_t i = 0; i < exp.we_wordc; i++) {
		// TODO - multiple silence recordings are not really supported yet!
		silence_output out(exp.we_wordv[i], output_directory);
		process_raw_audio_file(out);
	}
	wordfree(&exp);

	st = wordexp(fpattern.c_str(), &exp, WRDE_NOCMD | WRDE_SHOWERR | WRDE_UNDEF);
	if (st < 0)
		fatal("wordexp error");
	for (size_t i = 0; i < exp.we_wordc; i++) {
		dataset_output out(exp.we_wordv[i], output_directory);
		process_raw_audio_file(out);
	}
	wordfree(&exp);

	return EXIT_SUCCESS;
}
