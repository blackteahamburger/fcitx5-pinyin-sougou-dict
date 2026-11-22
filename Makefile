.PHONY: all
all: build

.PHONY: build
build: build_fcitx

.PHONY: build_fcitx
build_fcitx: sougou.dict

.PHONY: build_rime
build_rime: sougou.dict.yaml

sougou.source.fcitx:
	test -d sougou_dict || { echo The sougou_dict folder does not exist!; exit 1; }
	ImeWlConverterCmd -i scel -o libimetxt -O sougou.source.fcitx sougou_dict/*/*.scel

sougou.source.rime:
	test -d sougou_dict || { echo The sougou_dict folder does not exist!; exit 1; }
	ImeWlConverterCmd -i scel -o rime -O sougou.source.rime sougou_dict/*/*.scel

sougou.dict: sougou.source.fcitx
	libime_pinyindict sougou.source.fcitx sougou.dict

sougou.dict.yaml: sougou.source.rime
	printf -- '---\nname: sougou\nversion: "0.1"\nsort: by_weight\n...\n' > sougou.dict.yaml
	cat sougou.source.rime >> sougou.dict.yaml

.PHONY: install
install: install_fcitx

.PHONY: install_fcitx
install_fcitx: install_sougou_dict

.PHONY: install_rime
install_rime: install_sougou_dict_yaml

.PHONY: install_sougou_dict
install_sougou_dict: sougou.dict
	install -Dm644 sougou.dict -t $(DESTDIR)/usr/share/fcitx5/pinyin/dictionaries/

.PHONY: install_sougou_dict_yaml
install_sougou_dict_yaml: sougou.dict.yaml
	install -Dm644 sougou.dict.yaml -t $(DESTDIR)/usr/share/rime-data/

.PHONY: clean
clean:
	rm -f sougou.source.fcitx sougou.source.rime sougou.dict sougou.dict.yaml
