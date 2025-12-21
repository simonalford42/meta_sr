"""Print evolved operator functions from meta_evolution_results.json"""
import json

def main():
    with open("meta_evolution_results.json", "r") as f:
        results = json.load(f)

    for op_type in ["fitness", "selection", "mutation", "crossover"]:
        print("=" * 80)
        print(f" {op_type.upper()} OPERATOR ")
        print(f" Score: {results[op_type]['score']:.4f} | Lines: {results[op_type]['loc']}")
        print("=" * 80)
        print(results[op_type]["code"])
        print("\n")

if __name__ == "__main__":
    main()
