import heapq
import json


class Node:
    def __init__(
        self,
        probability: float,
        symbol: str = None,
        l_child: "Node" = None,
        r_child: "Node" = None,
    ):
        self.probability = probability
        self.symbol = symbol
        self.l_child = l_child
        self.r_child = r_child

    def __lt__(self, other: "Node") -> bool:
        return self.probability < other.probability


class FileHandler:
    @staticmethod
    def binfile_to_str(path: str) -> str:
        """Read binary data and return its bits as a string."""
        with open(path, "rb") as file:
            data = file.read()

        return "".join(f"{byte:08b}" for byte in data)

    @staticmethod
    def str_to_binfile(text: str, out_path: str) -> bool:
        """Write a string of 0s and 1s to a binary file."""
        try:
            if any(bit not in "01" for bit in text):
                raise ValueError("Input must contain only 0s and 1s.")

            padded = text.ljust((len(text) + 7) // 8 * 8, "0")

            data = bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))

            with open(out_path, "wb") as file:
                file.write(data)

            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def dict_to_json_file(data: dict, out_path: str) -> bool:
        """Save a dictionary to a JSON file."""
        try:
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

            return True
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def json_file_to_dict(path: str) -> dict:
        """Read a JSON file and return its contents as a dictionary."""
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("JSON file must contain an object (dictionary).")

        return data


class Huffman:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.pq = []
        self.other_symbols = []
        self.root = None
        self.codebook = {}

    def _calculate_freq(self, text: str) -> dict:
        """Calculate the frequency of each character."""
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        return freq

    def _calculate_probabilities(self, freq: dict, total_chars: int) -> dict:
        """Calculate the probability of each character."""
        return {char: count / total_chars for char, count in freq.items()}

    def _create_priority_queue(self, probabilities: dict) -> None:
        """Create a min-heap and group low-probability symbols into OTHER."""
        self.pq = []
        other_probability = 0.0
        self.other_symbols = []

        for symbol, probability in probabilities.items():
            if probability < self.threshold:
                other_probability += probability
                self.other_symbols.append(symbol)
            else:
                heapq.heappush(self.pq, Node(probability, symbol))

        if self.other_symbols:
            heapq.heappush(self.pq, Node(other_probability, "OTHER"))

    def _merge_smallest_nodes(self) -> None:
        """Merge the two smallest nodes and push their parent."""
        left = heapq.heappop(self.pq)
        right = heapq.heappop(self.pq)

        merged_node = Node(
            left.probability + right.probability,
            l_child=left,
            r_child=right,
        )
        heapq.heappush(self.pq, merged_node)

    def _build_tree(self) -> Node:
        """Build the Huffman tree."""
        while len(self.pq) > 1:
            self._merge_smallest_nodes()

        self.root = self.pq[0]
        return self.root

    def _generate_codes(self, node: Node, prefix: str = "") -> None:
        """Generate the codebook by traversing the tree."""
        if node.symbol is not None:
            self.codebook[node.symbol] = prefix or "0"
            return

        self._generate_codes(node.l_child, prefix + "0")
        self._generate_codes(node.r_child, prefix + "1")

    def _get_codes(self, text: str) -> dict:
        """Build the tree and generate its codebook."""
        if not text:
            self.pq = []
            self.root = None
            self.codebook = {}
            self.other_symbols = []
            return self.codebook

        freq = self._calculate_freq(text)
        probabilities = self._calculate_probabilities(freq, len(text))

        self.codebook = {}
        self._create_priority_queue(probabilities)
        self._build_tree()
        self._generate_codes(self.root)

        return self.codebook

    def compress(self, text: str, output_path: str = "compressed") -> str:
        codes = self._get_codes(text)

        other_codes = {
            symbol: format(i + 1, "04b") for i, symbol in enumerate(self.other_symbols)
        }

        compressed = "".join(
            codes[char] if char in codes else codes["OTHER"] + other_codes[char]
            for char in text
        )

        data = {
            "compressed_size": len(compressed),
            "codebook": codes,
            "other_symbols": other_codes,
        }

        FileHandler.dict_to_json_file(data, f"{output_path}_header.json")
        FileHandler.str_to_binfile(compressed, f"{output_path}.bin")

        return f"{output_path}.bin"

    def decompress(self, binary_path: str) -> str:
        header_path = binary_path.rsplit(".", 1)[0] + "_header.json"

        with open(header_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        bits = FileHandler.binfile_to_str(binary_path)
        bits = bits[: data["compressed_size"]]

        codebook = data["codebook"]
        other_codes = data["other_symbols"]

        # Reverse the mappings for decoding
        reverse_codebook = {code: symbol for symbol, code in codebook.items()}

        reverse_other_codes = {code: symbol for symbol, code in other_codes.items()}

        other_code = codebook["OTHER"]
        result = []
        buffer = ""
        i = 0

        while i < len(bits):
            buffer += bits[i]
            i += 1

            if buffer in reverse_codebook:
                symbol = reverse_codebook[buffer]
                buffer = ""

                if symbol == "OTHER":
                    extra_code = bits[i : i + 4]
                    i += 4

                    if extra_code not in reverse_other_codes:
                        raise ValueError(f"Invalid OTHER symbol code: {extra_code}")

                    result.append(reverse_other_codes[extra_code])
                else:
                    result.append(symbol)

        if buffer:
            raise ValueError("Invalid or incomplete Huffman bitstream.")

        return "".join(result)


import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress or decompress a file using Huffman coding."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="input path (text file to compress, or .bin file with -d)",
    )
    parser.add_argument(
        "-d",
        "--decompress",
        action="store_true",
        help="decompress the input .bin file instead of compressing",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="symbols with probability below this go into OTHER (default: 0.5, compress only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output path (default: <input name>.bin, or <input name>_restored.txt with -d)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    if args.decompress:
        return run_decompress(args)

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    return run_compress(args)


def run_compress(args) -> int:
    # compress() appends ".bin" and "_header.json" itself, so strip the extension
    output = args.output or os.path.splitext(args.file)[0]
    output = output.removesuffix(".bin")

    # newline="" keeps line endings exactly as they are in the file
    with open(args.file, "r", encoding="utf-8", newline="") as file:
        text = file.read()

    bin_path = Huffman(threshold=args.threshold).compress(text, output)

    original_size = os.path.getsize(args.file)
    compressed_size = os.path.getsize(bin_path)

    print(f"Compressed: {bin_path}")
    print(f"Header:     {output}_header.json")
    print(f"Size:       {original_size} -> {compressed_size} bytes")
    return 0


def run_decompress(args) -> int:
    header_path = args.file.rsplit(".", 1)[0] + "_header.json"
    if not os.path.isfile(header_path):
        print(f"Error: header file not found: {header_path}", file=sys.stderr)
        return 1

    try:
        text = Huffman().decompress(args.file)
    except (ValueError, KeyError) as error:
        print(f"Error: could not decompress: {error}", file=sys.stderr)
        return 1

    output = args.output or os.path.splitext(args.file)[0] + "_restored.txt"

    with open(output, "w", encoding="utf-8", newline="") as file:
        file.write(text)

    print(f"Decompressed: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
