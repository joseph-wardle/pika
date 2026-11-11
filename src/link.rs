//! Light and shadow linking: each distinct lightLink collection gets one bit of a
//! mask, and the light loop skips a light whose bit the mesh lacks; shadowLink
//! collections get bits of a second mask, and a light's shadows skip the meshes
//! lacking its bit.

use std::collections::BTreeMap;
use std::sync::Once;

use crate::store::{Light, Mesh, Store};

const BITS: usize = 32;

/// A light's bits, one or none per mask, or a mesh's: the collections it is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mask {
    pub light: u32,
    pub shadow: u32,
}

pub struct Links {
    light: BTreeMap<String, u32>,
    shadow: BTreeMap<String, u32>,
}

impl Links {
    pub fn new(store: &Store) -> Self {
        let links = || store.lights().values().filter_map(|l| l.link.as_ref());
        static LIGHT: Once = Once::new();
        static SHADOW: Once = Once::new();
        Self {
            light: bits(links().map(|l| l.light.as_str()), "light", &LIGHT),
            shadow: bits(links().map(|l| l.shadow.as_str()), "shadow", &SHADOW),
        }
    }

    /// Each bit 0 when the light reaches every mesh.
    pub fn light(&self, light: &Light) -> Mask {
        let bit = |bits: &BTreeMap<String, u32>, token: &str| bits.get(token).copied().unwrap_or(0);
        let link = light.link.as_ref();
        Mask {
            light: bit(&self.light, link.map_or("", |l| &l.light)),
            shadow: bit(&self.shadow, link.map_or("", |l| &l.shadow)),
        }
    }

    pub fn mesh(&self, mesh: &Mesh) -> Mask {
        let mask = |bits: &BTreeMap<String, u32>| {
            (mesh.categories.iter().flatten())
                .filter_map(|c| bits.get(c))
                .fold(0, |mask, bit| mask | bit)
        };
        Mask {
            light: mask(&self.light),
            shadow: mask(&self.shadow),
        }
    }
}

/// Bits in token order, so they do not hang on the order lights were added in.
/// Past `BITS` collections, a light reaches every mesh.
fn bits<'a>(
    tokens: impl Iterator<Item = &'a str>,
    kind: &str,
    warned: &Once,
) -> BTreeMap<String, u32> {
    let mut tokens: Vec<&str> = tokens.filter(|t| !t.is_empty()).collect();
    tokens.sort_unstable();
    tokens.dedup();
    if tokens.len() > BITS {
        warned.call_once(|| {
            eprintln!(
                "pika: warning: the lights {kind}-link {} collections, and Pika links {BITS}; the lights of the rest reach every mesh (warned once)",
                tokens.len()
            )
        });
    }
    (tokens.into_iter().take(BITS).enumerate())
        .map(|(i, t)| (t.to_owned(), 1 << i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Link;

    /// As the light loop decides it, and as shadows will.
    fn reaches(light: u32, mesh: u32) -> bool {
        light & mesh == light
    }

    fn store(links: &[&str], categories: &[&[&str]]) -> (Store, Vec<u64>, Vec<u64>) {
        let mut store = Store::new();
        let lights = (links.iter().enumerate())
            .map(|(i, link)| {
                let id = store.insert_light(format!("/light{i}"));
                let link = Link {
                    light: (*link).into(),
                    shadow: String::new(),
                };
                store.set_light_link(id, link).unwrap();
                id
            })
            .collect();
        let meshes = (categories.iter().enumerate())
            .map(|(i, c)| {
                let id = store.insert_mesh(format!("/mesh{i}"));
                let c = c.iter().map(|&t| t.to_owned()).collect();
                store.set_categories(id, c).unwrap();
                id
            })
            .collect();
        (store, lights, meshes)
    }

    fn lit(store: &Store, lights: &[u64], meshes: &[u64]) -> Vec<Vec<bool>> {
        let links = Links::new(store);
        (meshes.iter())
            .map(|m| {
                let mesh = links.mesh(&store.meshes()[m]).light;
                (lights.iter())
                    .map(|l| reaches(links.light(&store.lights()[l]).light, mesh))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn lights_light_the_meshes_of_their_collection() {
        let (store, lights, meshes) = store(
            &["group_1", "", "group_0", "group_1"],
            &[&["group_0"], &["group_1"], &[], &["group_0", "group_1"]],
        );
        let t = true;
        let f = false;
        assert_eq!(
            lit(&store, &lights, &meshes),
            [[f, t, t, f], [t, t, f, t], [f, t, f, f], [t, t, t, t]]
        );
    }

    #[test]
    fn categories_no_light_links_light_nothing_more() {
        let (store, lights, meshes) = store(&["group_0", ""], &[&["group_9"], &[]]);
        assert_eq!(
            lit(&store, &lights, &meshes),
            [[false, true], [false, true]]
        );
        // A mesh not yet synced is in no collection.
        let mut store = store;
        let unsynced = store.insert_mesh("/unsynced".into());
        assert_eq!(
            Links::new(&store).mesh(&store.meshes()[&unsynced]),
            Mask::default()
        );
    }

    #[test]
    fn bits_follow_tokens_not_light_order() {
        let (a, la, _) = store(&["b", "a"], &[]);
        let (b, lb, _) = store(&["a", "b"], &[]);
        let bit = |s: &Store, id: u64| Links::new(s).light(&s.lights()[&id]).light;
        assert_eq!(bit(&a, la[0]), bit(&b, lb[1]));
        assert_ne!(bit(&a, la[0]), bit(&a, la[1]));
    }

    #[test]
    fn lights_past_the_last_bit_light_every_mesh() {
        let tokens: Vec<String> = (0..=BITS).map(|i| format!("group_{i:02}")).collect();
        let links: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let (store, lights, meshes) = store(&links, &[&[]]);
        let lit = &lit(&store, &lights, &meshes)[0];
        assert_eq!(lit.iter().filter(|&&l| l).count(), 1);
        assert!(lit[BITS]);
    }

    /// A light's shadow link picks its casters apart from what it lights: the sun lights
    /// both meshes, but only the one in its shadowLink collection casts.
    #[test]
    fn shadow_links_pick_casters_apart_from_light_links() {
        let link = |light: &str, shadow: &str| Link {
            light: light.into(),
            shadow: shadow.into(),
        };
        let tokens = |t: &[&str]| t.iter().map(|&t| t.to_owned()).collect();
        let mut store = Store::new();
        let sun = store.insert_light("/sun".into());
        let key = store.insert_light("/key".into());
        let caster = store.insert_mesh("/caster".into());
        let window = store.insert_mesh("/window".into());
        store.set_light_link(sun, link("", "group_1")).unwrap();
        store.set_light_link(key, link("group_0", "")).unwrap();
        store
            .set_categories(caster, tokens(&["group_0", "group_1"]))
            .unwrap();
        store.set_categories(window, tokens(&["group_0"])).unwrap();
        let masks = |store: &Store| {
            let links = Links::new(store);
            let light = |id| links.light(&store.lights()[&id]);
            let mesh = |id| links.mesh(&store.meshes()[&id]);
            (light(sun), light(key), mesh(caster), mesh(window))
        };
        let (s, k, c, w) = masks(&store);
        assert!(reaches(s.light, c.light) && reaches(s.light, w.light));
        assert!(reaches(s.shadow, c.shadow));
        assert!(
            !reaches(s.shadow, w.shadow),
            "the window casts the sun's shadow"
        );
        assert!(reaches(k.light, w.light) && reaches(k.shadow, w.shadow));
        // Negative control: with the link removed, the window casts the sun's shadow.
        store.set_light_link(sun, link("", "")).unwrap();
        let (s, _, _, w) = masks(&store);
        assert!(reaches(s.shadow, w.shadow));
    }

    #[test]
    fn links_move_the_draws() {
        let (mut store, lights, meshes) = store(&["group_0"], &[&[]]);
        store.take_moved();
        let link = |light: &str, shadow: &str| Link {
            light: light.into(),
            shadow: shadow.into(),
        };
        store
            .set_light_link(lights[0], link("group_0", ""))
            .unwrap();
        store.set_categories(meshes[0], Vec::new()).unwrap();
        assert!(!store.take_moved().draws, "an unchanged link moves nothing");
        store
            .set_light_link(lights[0], link("group_1", ""))
            .unwrap();
        assert!(store.take_moved().draws);
        store
            .set_light_link(lights[0], link("group_1", "group_2"))
            .unwrap();
        assert!(store.take_moved().draws, "a shadow link moves the casters");
        store
            .set_categories(meshes[0], vec!["group_1".into()])
            .unwrap();
        assert!(store.take_moved().draws);
        store.remove_light(lights[0]).unwrap();
        assert!(store.take_moved().draws);
    }
}
