> kill -9 matching pattern when parsed with 2nd entry (space separated)

```
kill -9  $(ps -ef | grep 'python' | grep 'test' | awk '{print $2}')
```

