git ls-files '*.urdf' '*.xacro' '*.srdf' | while read f; do
    xmllint --format "$f" -o "$f.tmp" && mv "$f.tmp" "$f"
done
