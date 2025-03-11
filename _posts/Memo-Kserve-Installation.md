
- knative-serving PeerAuthentication 설정

```bash
# 적용
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ cat <<EOF | kubectl apply -f -
apiVersion: "security.istio.io/v1beta1"
kind: "PeerAuthentication"
metadata:
  name: "default"
  namespace: "knative-serving"
spec:
  mtls:
    mode: PERMISSIVE
EOF
peerauthentication.security.istio.io/default created
# 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get peerauthentication -n knative-serving
NAME                MODE         AGE
default             PERMISSIVE   47s
net-istio-webhook                2d22h
webhook                          2d22h
```