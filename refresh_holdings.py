import json

from tools import refresh_treasury_holdings_tool


def main() -> None:
    result = refresh_treasury_holdings_tool()
    payload = json.loads(result)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
