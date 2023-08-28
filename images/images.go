// Package images embeds all images used by the game, to avoid external dependencies.
package images

import (
	_ "embed"
	"github.com/pkg/errors"
	"os"
	"path"
)

//go:embed Ant.png
var AntBytes []byte

//go:embed Beetle.png
var BeetleBytes []byte

//go:embed Grasshopper.png
var GrasshopperBytes []byte

//go:embed Queen.png
var QueenBytes []byte

//go:embed Spider.png
var SpiderBytes []byte

//go:embed Icon.png
var IconBytes []byte

//go:embed colors.png
var ColorsBytes []byte

//go:embed tile_player_0.png
var Tile0Bytes []byte

//go:embed tile_player_1.png
var Tile1Bytes []byte

type Info struct {
	Name  string
	Bytes []byte
}

var NameToInfo map[string]Info

func init() {
	NameToInfo = make(map[string]Info)
	for _, info := range []Info{
		{"Ant", AntBytes},
		{"Beetle", BeetleBytes},
		{"Grasshopper", GrasshopperBytes},
		{"Queen", QueenBytes},
		{"Spider", SpiderBytes},
		{"Icon", IconBytes},
		{"colors", ColorsBytes},
		{"tile_player_0", Tile0Bytes},
		{"tile_player_1", Tile1Bytes},
	} {
		NameToInfo[info.Name] = info
	}
}

var (
	// TmpDir is a temporary directory where the images are materialized to disk,
	// if CopyToTmpDir is called.
	TmpDir string
)

// CopyToTmpDir copies the embedded images to a temporary directory.
//
// This seems counter-productive, since they are going to be read again, but the
// Cairo (part of gnome) library only takes as an input a filename :( -- or another
// more complicated way.
//
// It caches the name of directory created, and if it's called again, it returns
// the cached value.
func CopyToTmpDir() (string, error) {
	if TmpDir != "" {
		return TmpDir, nil
	}
	dir, err := os.MkdirTemp("", "hiveGoImage")
	if err != nil {
		return "", errors.Wrapf(err, "failed to create temporary directory to hold game images")
	}
	TmpDir = dir // Cache result.

	// Copy over images.
	for name, info := range NameToInfo {
		p := info.Path()
		err := os.WriteFile(p, info.Bytes, 0666)
		if err != nil {
			err = errors.Wrapf(err, "failed to copy image %q to images tmp dir %q", name, TmpDir)
			TmpDir = "" // Reset cache value, since not all images were written.
			return "", err
		}
	}

	return dir, nil
}

// Path returns the path to a copy of the image, created by CopyToTmpDir.
// If `info` is empty or if CopyToTmpDir was not called or failed, this returns
// an empty string.
func (info Info) Path() string {
	if TmpDir == "" {
		return ""
	}
	if info.Name == "" {
		return ""
	}
	return path.Join(TmpDir, info.Name+".png")
}
