```java
    /**
     * l in [0-100], a in [-128,128], b in [-128,128]
     *
     * @param c input color
     * @return corresponding lab color
     */
    public static double[] rgb2lab(Color c) {
        double R = gamma(c.getRed() / 255.0);
        double G = gamma(c.getGreen() / 255.0);
        double B = gamma(c.getBlue() / 255.0);

        double[] xyz = new double[3];
        xyz[0] = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.95047;
        xyz[1] = (0.212671 * R + 0.715160 * G + 0.072169 * B) / 1.0;
        xyz[2] = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.08883;

        double[] lab = new double[3];
        lab[0] = xyz[1] > 0.008856 ? (116.0 * f(xyz[1]) - 16.0) : (903.3 * xyz[1]);
        lab[1] = 500.0 * (f(xyz[0]) - f(xyz[1]));
        lab[2] = 200.0 * (f(xyz[1]) - f(xyz[2]));

        return lab;
    }

    private static double gamma(double x) {
        return x > 0.04045 ? pow((x + 0.055) / 1.055, 2.4) : x / 12.92;
    }

    private static double f(double x) {
        return x > 0.008856 ? pow(x, 1.0 / 3.0) : (7.787 * x + 0.137931);
    }
```
