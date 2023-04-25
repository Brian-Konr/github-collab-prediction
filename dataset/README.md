# 資料

json 寫的是要的欄位

## User

### User Profile
Endpoint: /users/{username}

```json
{
  "login": "octocat",
  "name": "The Octocat",
  "company": "GitHub",
  "bio": "There once was...",
  "location": "Taipei, Taiwan",
  "email": "octocat@github.com"
}
```

### User follower
Endpoint: /users/{username}/followers
應該是用 login 來代表一個 user 所以應該只需要 login
```json
[
    {
        "login": "octocat",
        "name": "The Octocat"
    }, ...
]
```

### User following
Endpoint: /users/{username}/following
```json
[
    {
        "login": "octocat",
        "name": "The Octocat"
    }, ...
]
```

### Starred Repositories
Endpoint: /users/{username}/starred

需在Header設定: Accept 為 application/vnd.github.star+json

description,full_name, starred_at 是需要的資料, id 只是看爬蟲會不會更方便處理
```json
[
  {
    "starred_at": "2012-10-09T23:39:01Z",
    "repo": {
      "id": 1296269,
      "full_name": "octocat/Hello-World",
      "html_url": "https://github.com/twitter/twitter",
      "description": "This your first repo!"
    }
  }
]
```

### Repo Languages
Endpoint: /repos/{owner}/{repo}/languages
```json
{
  "C": 78769,
  "Python": 7769
}
```

### List Repo
Endpoint: /users/{username}/repos?type=all

可以得到用戶所有有關的 Repo

```json
[
  {
    "id": 1296269,
    "name": "Hello-World",
    "full_name": "octocat/Hello-World",
    "owner": {
      "login": "octocat"
    },
    "url": "https://api.github.com/repos/octocat/Hello-World"
  }
]
```

### Repo Commits
Endpoint: /repos/{owner}/{repo}/commits

需在Header設定: Accept 為 application/vnd.github.star+json

Query: per_page=100 page=1
```json
[
  {
    "commit": {
      "author": {
        "name": "Monalisa Octocat",
        "email": "octocat@github.com",
        "date": "2011-04-14T16:00:49Z"
      }
    }
  }
]
```

### Process
In Psuedocode

