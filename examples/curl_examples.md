# API Examples

## Linux / macOS

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json; charset=utf-8" \
     --data-binary @examples/basic_input.json
```

## Windows (PowerShell)

```powershell
curl -X POST "http://127.0.0.1:8000/summarize" `
     -H "Content-Type: application/json; charset=utf-8" `
     --data-binary "@examples/basic_input.json"
```

## Windows (CMD)

```cmd
curl -X POST "http://127.0.0.1:8000/summarize" ^
     -H "Content-Type: application/json; charset=utf-8" ^
     --data-binary "@examples/input_hebrew.json"
```

