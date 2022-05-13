# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import statistics
import time
import tempfile
from typing import Callable, List, Mapping, NamedTuple, Sequence
import os
import pathlib
import subprocess
import wave


INPUT_FILES = [
    "testwav/wikimedia.i_love_you_california.wav",
    "testwav/wikimedia.winter_kiss.wav",
    "testwav/wikimedia.jazz_funk_no1_sax.wav",
]
REPORT_OUTPUT = "report"
REFERENCE_ENCODER_OPTS = {
    # "opt9lax": ["-f", "-9", "--lax"],  # This requires a customized binary.
    "opt8lax": ["-f", "-8", "--lax"],
    "opt8": ["-f", "-8"],
    "opt5": ["-f", "-5"],
}
TEST_ENCODER_OPTS = {
    "default": [],
    "dmse": ["-c", "report/dmse.config.toml"],
}
REFERENCE_BINPATH = "flac-1.3.4/src/flac/flac"


def logged(f):
    def wrapped_f(*args, **kwargs):
        print(f"Calling {args[0]}")
        return f(*args, **kwargs)

    return wrapped_f


class EncoderRunStat(NamedTuple):
    input_duration: float
    time: float
    input: pathlib.Path
    input_size: int
    output: pathlib.Path
    output_size: int

    @property
    def comparession_rate(self):
        return self.output_size / self.input_size

    @property
    def rtf(self):
        return self.time / self.input_duration


SourceNameToRunStat = Mapping[str, EncoderRunStat]
RunResults = Mapping[str, SourceNameToRunStat]


def run_encoder(
    inputs: Sequence[pathlib.Path],
    output_root: pathlib.Path,
    encoder_bin: str,
    encoder_opts: Mapping[str, List[str]],
) -> RunResults:
    results = {optname: dict() for optname in encoder_opts.keys()}

    for inp in inputs:
        input_wav = wave.open(str(inp), "rb")
        duration = input_wav.getnframes() / input_wav.getframerate()
        print(f"duration={duration}")

        for optname, opts in encoder_opts.items():
            output_path = output_root / inp.stem / f"{optname}.flac"
            if not output_path.parent.is_dir():
                os.makedirs(output_path.parent)

            start = time.time()
            logged(subprocess.check_call)(
                [encoder_bin] + opts + ["-o", str(output_path), str(inp)]
            )
            results[optname][inp.stem] = EncoderRunStat(
                input_duration=duration,
                time=time.time() - start,
                input=inp,
                input_size=inp.stat().st_size,
                output=output_path,
                output_size=output_path.stat().st_size,
            )

    return results


def assert_eq(expected, actual):
    if expected != actual:
        expected_disp = str(expected)
        actual_disp = str(actual)
        if len(expected_disp) > 80:
            expected_disp = (
                expected_disp[:10] + "<snipped>" + expected_disp[-10:]
            )
        if len(actual_disp) > 80:
            actual_disp = actual_disp[:10] + "<snipped>" + actual_disp[-10:]
        raise ValueError(f"{expected_disp} != {actual_disp}")


def verify_with_decoder(encoder_run_results: RunResults, decoder_bin: str):
    for unused_confname, results in encoder_run_results.items():
        for unused_sourcename, runstat in results.items():
            with tempfile.NamedTemporaryFile() as tmpout:
                logged(subprocess.check_call)(
                    [
                        decoder_bin,
                        "-f",
                        "-d",
                        "-o",
                        tmpout.name,
                        str(runstat.output),
                    ]
                )
                ref = wave.open(str(runstat.input), "rb")
                decoded = wave.open(tmpout.name, "rb")

                assert_eq(ref.getframerate(), decoded.getframerate())
                assert_eq(ref.getsampwidth(), decoded.getsampwidth())
                assert_eq(ref.getnchannels(), decoded.getnchannels())
                assert_eq(ref.getnframes(), decoded.getnframes())

                assert_eq(
                    ref.readframes(ref.getnframes()),
                    decoded.readframes(decoded.getnframes()),
                )


def itemize(vals: Sequence[str]):
    ret = ""
    for v in vals:
        ret += f"- {v}"
    return ret


def itemize_average(
    run_results: RunResults, *, key: Callable[[EncoderRunStat], float]
):
    ret = ""
    for confname, results in run_results.items():
        vals = [key(runstat) for unuserd_srcname, runstat in results.items()]
        ret += f"- {confname}: {statistics.mean(vals)}\n"
    return ret


def itemize_compression_rate(run_results: RunResults):
    return itemize_average(run_results, key=lambda x: x.comparession_rate)


def itemize_inverse_rtf(run_results: RunResults):
    return itemize_average(run_results, key=lambda x: 1.0 / x.rtf)


def indent(indent: int, s: str):
    ret = ""
    for line in s.splitlines():
        ret += (" " * indent) + line + "\n"
    return ret


def make_report(
    ref_run_results: RunResults,
    test_run_results: RunResults,
    output_path: pathlib.Path,
):
    sources = list(test_run_results["default"].keys())
    with open(output_path, "w") as dest:
        print(
            f"""
# Encoder Comparison Report

## Summary

Sources used: {', '.join(sources)}

### Average compression rate

  - Reference
{indent(4, itemize_compression_rate(ref_run_results))}
  - Ours
{indent(4, itemize_compression_rate(test_run_results))}

### Average compression speed (inverse RTF)
  - Reference
{indent(4, itemize_inverse_rtf(ref_run_results))}
  - Ours
{indent(4, itemize_inverse_rtf(test_run_results))}
""",
            file=dest,
        )


def main():
    project_root = pathlib.Path(__file__).parent.parent
    inputs = [project_root / pathlib.Path(p) for p in INPUT_FILES]
    report_root = project_root / pathlib.Path(REPORT_OUTPUT)
    refenc_out_root = report_root / "refenc_out"
    testenc_out_root = report_root / "testenc_out"
    report_md_out = report_root / "report.md"

    # build
    logged(subprocess.check_call)(
        ["bash", "-c", f"cd {project_root}; cargo build --release"]
    )
    print("Running reference encoder.")
    ref_run_results = run_encoder(
        inputs, refenc_out_root, REFERENCE_BINPATH, REFERENCE_ENCODER_OPTS
    )
    print("Running test encoder.")
    test_run_results = run_encoder(
        inputs,
        testenc_out_root,
        str(project_root / "target" / "release" / "flacenc-rs"),
        TEST_ENCODER_OPTS,
    )

    print(ref_run_results)
    print(test_run_results)

    print("Checking output.")
    verify_with_decoder(test_run_results, REFERENCE_BINPATH)
    print("Making report.")
    make_report(ref_run_results, test_run_results, report_md_out)


if __name__ == "__main__":
    main()
