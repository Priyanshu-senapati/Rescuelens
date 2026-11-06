# Where the high-voltage information comes from

Every `battery` and `hv_cables` line in [`data/hazards.json`](../data/hazards.json) was read from the document below, not written
from memory. The app shows these lines to someone deciding where to cut, so:

- **Do not edit a location without reading the guide.** If a guide is missing, set the field to `null`.
- **Page numbers count PDF pages**, not the number printed on the page.

| Vehicle | Document (publisher) | Hosted by | Battery | Cables |
|---|---|---|---|---|
| Tesla Model S 2012 | [2012-13 Model S Emergency Response Guide](https://digitalassets.tesla.com/tesla-contents/image/upload/2012-13_Model_S_Emergency_Response_Guide_en.pdf) (Tesla) | Tesla | p.6, p.21, p.23 | p.8 |
| Nissan Leaf 2012 | [2012 LEAF Dismantling Guide](https://owners.nissanusa.com/content/techpub/ManualsAndGuides/LEAF/2012/2012-leaf-dismantling-guide.pdf) (Nissan) | Nissan | p.5, p.10 | p.10-11 |
| Chevrolet Tahoe Hybrid 2012, GMC Yukon Hybrid 2012, Chevrolet Silverado 1500 Hybrid 2012 | [Two-mode vehicles First Responder Guide](https://morongocsd.specialdistrict.org/files/847ddc729/Chevrolet+Tahoe+HEV+2008-2013+ERG.pdf) (GM Service Technical College) | Morongo Valley fire district | p.16, p.23 | p.11, p.15, p.24 |
| Chevrolet Malibu Hybrid 2010 | [Malibu Hybrid and 2007 Aura Green Line Emergency Response Guide](https://boronextrication.com/wp-content/uploads/2010/11/Chevrolet-Malibu-2007-Saturn-AURA-Green-Line-Hybrid.pdf) (GM, 2007) | Boron Extrication | p.18 | p.15, p.24 |
| BMW ActiveHybrid 5 2012 | [5er-Reihe ActiveHybrid 5 F10 rescue sheet, German, edition 11/2012](https://ff.mts-online.com/data/pdf/BMW_5er_Serie_F10_2011_Active_Hybrid.pdf) (BMW AG) | MTS | p.2, p.5 | p.3-6 |
| Hyundai Sonata Hybrid 2012 | [LF Sonata HEV Emergency Response Guide](https://boronextrication.com/wp-content/uploads/2010/11/LF-SONATA-HEV-Emergency-Response-Guide.pdf) (Hyundai) | Boron Extrication | p.11 | p.6-7, p.30 |
| Fisker Karma 2012 | [High Voltage Safety and First Response Info Guide, Nov 2011](https://www.morongovalleyfire.org/files/14cfd88f9/Fisker+Karma+PHEV+2011-2012+ERG.pdf) (Fisker Automotive) | Morongo Valley fire department | p.4 diagram, p.27 | p.4, p.16 |

## Caveats to say out loud in the viva

1. **Seven of the nine links are copies.** Only Tesla and Nissan are served by the manufacturer. The others were published by the
   manufacturer, but the copies we could reach are hosted by fire services, a rescue-training site and a rescue-software vendor.
   Any of those links could go away.
2. **The Malibu guide predates the 2010 car.** It is GM's guide for the Malibu Hybrid, copyright 2007, and the closest we found. Its
   cables are **blue, 36-42 V**, not orange, so the app's standard advice now mentions blue.
3. **The BMW sheet is in German**; the app's text is our translation of "Die Hochvolt-Batterie befindet sich unter der
   Gepäckraumverkleidung". A native speaker should check it.
4. **The Fisker battery route comes from a diagram**, not a sentence: the guide's do-not-cut image shows the pack along the tunnel.
5. **The Leaf link is Nissan's dismantling guide**, whose responder sections match the 2012 First Responder's Guide page for page.
   We used it because Nissan hosts it; the First Responder's Guide itself only turned up on a third-party site.
6. **"Passenger side" means US left-hand-drive cars.** These are US-market guides, and Stanford Cars is a US-market dataset.
7. **None of this has been reviewed by a responder.** It is a prototype, and the app says so on every answer.
