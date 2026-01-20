def compute_masked_percentage(image, aoi, scale=10):
    band = image.select('B4')

    # 1 = valid, 0 = masked
    valid = band.mask().rename('valid')

    # constant image for total pixels
    total = ee.Image.constant(1).rename('total')

    stats = ee.Image.cat([valid, total]).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e9
    )

    valid_pixels = ee.Number(stats.get('valid'))
    total_pixels = ee.Number(stats.get('total'))

    masked_pixels = total_pixels.subtract(valid_pixels)
    masked_percent = masked_pixels.divide(total_pixels).multiply(100)

    return masked_pixels, masked_percent, total_pixels
